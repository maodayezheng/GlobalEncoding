import tensorflow as tf


class GRUCell(object):
    def __init__(self, input_dim, hidden_dim):
        self.hid_dim = hidden_dim
        self.input_dim = input_dim
        self.gru_gates_w = tf.Variable(tf.random_uniform([input_dim+hidden_dim, 2*hidden_dim], minval=-0.05, maxval=0.05))
        self.gru_gates_b = tf.Variable(tf.random_uniform([2*hidden_dim, ], minval=-0.05, maxval=0.05))
        self.gru_candidate_w = tf.Variable(tf.random_uniform([input_dim+hidden_dim, hidden_dim], minval=-0.05, maxval=0.05))
        self.gru_candidate_b = tf.Variable(tf.random_uniform([hidden_dim,], minval=-0.05, maxval=0.05))

    def __call__(self, x, h):
        rnn_in = tf.concat(1, [x, h])
        gates = tf.nn.sigmoid(tf.matmul(rnn_in, self.gru_gates_w) + tf.reshape(self.gru_gates_b, shape=[1, self.hid_dim*2]))
        r = gates[:, :self.hid_dim]
        u = gates[:, self.hid_dim:]
        rnn_in = tf.concat(1, [x, h*r])
        candidate = tf.tanh(tf.matmul(rnn_in, self.gru_candidate_w) + tf.reshape(self.gru_candidate_b, shape=[1, self.hid_dim]))
        output = (1.0 - u) * candidate + u * candidate
        return output