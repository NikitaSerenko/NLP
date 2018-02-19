import tensorflow as tf
import keras.layers as L

from basic_model import infer_length


# put all attention into model, because it needs only one function and 1-3 variables, easier to do model+att at once
class AttentionGRUTranslationModel:
    def __init__(self, name, inp_voc, out_voc,
                 emb_size, hid_size, attention_fun='bilinear', bidirectional=False, init='zero'):

        self.name = name
        self.att_func = attention_fun
        self.inp_voc = inp_voc
        self.out_voc = out_voc
        
        self.bidirectional = bidirectional

        with tf.variable_scope(name):
            self.emb_inp = L.Embedding(len(inp_voc), emb_size)
            self.emb_out = L.Embedding(len(out_voc), emb_size)
            if bidirectional:
                self.enc0 = tf.nn.rnn_cell.GRUCell(hid_size // 2)
                self.enc1 = tf.nn.rnn_cell.GRUCell(hid_size // 2)
            else:
                self.enc0 = tf.nn.rnn_cell.GRUCell(hid_size)
            self.dec_start = L.Dense(hid_size)
            self.dec0 = tf.nn.rnn_cell.GRUCell(hid_size)
            self.logits = L.Dense(len(out_voc))
            
            if self.att_func != 'simple':
                if self.att_func == 'tanh_hard':
                    self.att_W_1 = tf.Variable(tf.random_normal((hid_size, hid_size), stddev=0.1))
                    self.att_W_2 = tf.Variable(tf.random_normal((hid_size, hid_size), stddev=0.1))
                    self.att_v = tf.Variable(tf.random_normal((hid_size,), stddev=0.1))
                else:
                    if self.att_func == 'tanh':
                        self.att_u_omega = tf.Variable(tf.random_normal([1], stddev=0.1))
                    if init == 'zero':
                        self.att_W = tf.Variable(tf.zeros((hid_size, hid_size)))
                    else:
                        self.att_W = tf.Variable(tf.random_normal((hid_size, hid_size), stddev=0.1))


            # run on dummy output to .build all layers (and therefore create weights)
            inp = tf.placeholder('int32', [None, None])
            out = tf.placeholder('int32', [None, None])
            outputs, h0 = self.encode(inp)
            att = self._attention(outputs, h0)
            h1 = self.decode(h0, out[:,0], att)
            # h2 = self.decode(h1,out[:,1]) etc.

        self.weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)

    def _attention(self, encodings, prev_hidden):
        if self.att_func == 'simple':  # just muptiply vectors
            scores = tf.reduce_sum(tf.transpose(encodings, [1, 0, 2]) * prev_hidden, axis=2)
        elif self.att_func in ('bilinear', 'tanh'):
            scores = tf.tensordot(encodings, self.att_W, axes=[[2], [0]])
            scores = tf.transpose(scores, [1, 0, 2]) 
            scores = tf.reduce_sum(scores * prev_hidden, axis=2)
            if self.att_func == 'tanh':  # add tanh and scalar before output
                scores = tf.tanh(scores) * self.att_u_omega
        elif self.att_func == 'tanh_hard':
            scores = tf.tensordot(encodings, self.att_W_1, axes=[[2], [0]])
            scores = tf.transpose(scores, [1, 0, 2]) 
            scores_prev_hidden = tf.tensordot(prev_hidden, self.att_W_2, axes=[[1], [0]])
            
            scores = scores + scores_prev_hidden
            scores = tf.tensordot(tf.tanh(scores), self.att_v, axes=[[2], [0]])

        probs = tf.nn.softmax(scores, dim=0)
        attention = tf.reduce_sum(probs * tf.transpose(encodings, [2, 1, 0]), axis=1)
        attention = tf.transpose(attention, [1, 0])
        return attention

    def encode(self, inp, **flags):
        """
        Takes symbolic input sequence, computes initial state
        :param inp: matrix of input tokens [batch, time]
        :return: a list of initial decoder state tensors
        """
        inp_lengths = infer_length(inp, self.inp_voc.eos_ix)
        inp_emb = self.emb_inp(inp)

        if self.bidirectional:
            outputs, states = tf.nn.bidirectional_dynamic_rnn(
                self.enc0, self.enc1, inp_emb,
                sequence_length=inp_lengths,
                dtype=inp_emb.dtype)
            return tf.concat(outputs, 2), tf.concat(states, 1)
        else:
            _, enc_last = tf.nn.dynamic_rnn(
                              self.enc0, inp_emb,
                              sequence_length=inp_lengths,
                              dtype = inp_emb.dtype)

#             dec_start = self.dec_start(enc_last)
            return _, enc_last

    def decode(self, prev_state, prev_tokens, attention, **flags):
        """
        Takes previous decoder state and tokens, returns new state and logits
        :param prev_state: a list of previous decoder state tensors
        :param prev_tokens: previous output tokens, an int vector of [batch_size]
        :return: a list of next decoder state tensors, a tensor of logits [batch,n_tokens]
        """

#         [prev_dec] = prev_state

        prev_emb = self.emb_out(prev_tokens[:,None])[:,0]

        new_dec_out, new_dec_state = self.dec0(tf.concat((prev_emb, attention), axis=1), prev_state)

        output_logits = self.logits(new_dec_out)

        return [new_dec_state], output_logits

    def symbolic_score(self, inp, out, eps=1e-30, **flags):
        """
        Takes symbolic int32 matrices of hebrew words and their english translations.
        Computes the log-probabilities of all possible english characters given english prefices and hebrew word.
        :param inp: input sequence, int32 matrix of shape [batch,time]
        :param out: output sequence, int32 matrix of shape [batch,time]
        :return: log-probabilities of all possible english characters of shape [bath,time,n_tokens]

        NOTE: log-probabilities time axis  is synchronized with out
        In other words, logp are probabilities of __current__ output at each tick, not the next one
        therefore you can get likelihood as logprobas * tf.one_hot(out,n_tokens)
        """
        encodings, first_state = self.encode(inp,**flags)

        batch_size = tf.shape(inp)[0]
        bos = tf.fill([batch_size],self.out_voc.bos_ix)
        first_logits = tf.log(tf.one_hot(bos, len(self.out_voc)) + eps)

        def step(blob, y_prev):
            h_prev = blob[0]
            att = self._attention(encodings, h_prev)
            h_new, logits = self.decode(h_prev, y_prev, att, **flags)
            return list(h_new) + [logits]

        results = tf.scan(step,initializer=[first_state]+[first_logits],
                          elems=tf.transpose(out))

        # gather state and logits, each of shape [time,batch,...]
        states_seq, logits_seq = results[:-1], results[-1]

        # add initial state and logits
        logits_seq = tf.concat((first_logits[None], logits_seq),axis=0)
       
        #convert from [time,batch,...] to [batch,time,...]
        logits_seq = tf.transpose(logits_seq, [1, 0, 2])
       
        return tf.nn.log_softmax(logits_seq)

    def symbolic_translate(self, inp, greedy=False, max_len = None, eps = 1e-30, **flags):
        """
        takes symbolic int32 matrix of hebrew words, produces output tokens sampled
        from the model and output log-probabilities for all possible tokens at each tick.
        :param inp: input sequence, int32 matrix of shape [batch,time]
        :param greedy: if greedy, takes token with highest probablity at each tick.
            Otherwise samples proportionally to probability.
        :param max_len: max length of output, defaults to 2 * input length
        :return: output tokens int32[batch,time] and
                 log-probabilities of all tokens at each tick, [batch,time,n_tokens]
        """
        encodings, first_state = self.encode(inp, **flags)

        batch_size = tf.shape(inp)[0]
        bos = tf.fill([batch_size],self.out_voc.bos_ix)
        first_logits = tf.log(tf.one_hot(bos, len(self.out_voc)) + eps)
        max_len = tf.reduce_max(tf.shape(inp)[1])*2

        def step(blob,t):
            h_prev, y_prev = blob[0], blob[-1]
            att = self._attention(encodings, h_prev)
            h_new, logits = self.decode(h_prev, y_prev, att, **flags)
            y_new = tf.argmax(logits,axis=-1) if greedy else tf.multinomial(logits,1)[:,0]
            return list(h_new) + [logits, tf.cast(y_new,y_prev.dtype)]

        results = tf.scan(step, initializer=[first_state] + [first_logits, bos],
                          elems=[tf.range(max_len)])

        # gather state, logits and outs, each of shape [time,batch,...]
        states_seq, logits_seq, out_seq = results[:-2], results[-2], results[-1]

        # add initial state, logits and out
        logits_seq = tf.concat((first_logits[None],logits_seq),axis=0)
        out_seq = tf.concat((bos[None], out_seq), axis=0)
        
        #convert from [time,batch,...] to [batch,time,...]
        logits_seq = tf.transpose(logits_seq, [1, 0, 2])
        out_seq = tf.transpose(out_seq)
        return out_seq, tf.nn.log_softmax(logits_seq)
