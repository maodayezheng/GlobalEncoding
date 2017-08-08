import tensorflow as tf
import json
import numpy as np
import os
import time
from tensorflow.contrib.distributions import Categorical
from Cells import GRUCell


class RNNBuilder(object):
    def __init__(self, vocab_size, en_embed_dim, de_embed_dim, hid_dim):
        self.vocab_size = vocab_size
        self.en_embed_dim = en_embed_dim
        self.de_embed_dim = de_embed_dim
        self.hid_dim = hid_dim

        # Init word embedding
        self.input_embedding = tf.Variable(tf.random_uniform([vocab_size, de_embed_dim], minval=-0.05, maxval=0.05))
        self.output_embedding = tf.Variable(tf.random_uniform([vocab_size, de_embed_dim], minval=-0.05, maxval=0.05))

        # Init RNN cell
        self.cell = GRUCell(en_embed_dim, hid_dim)

        # Output projection layer
        self.output_w = tf.Variable(tf.random_uniform([en_embed_dim + hid_dim, de_embed_dim], minval=-0.05, maxval=0.05))
        self.output_b = tf.Variable(tf.random_uniform([de_embed_dim, ], minval=-0.05, maxval=0.05))

    def build_training_graph(self, max_l):
        source = tf.placeholder(dtype=tf.int32, name="source", shape=[None, max_l])
        target = tf.placeholder(dtype=tf.int32, name="target", shape=[None, max_l])

        # Get model parameters
        input_embedding = self.input_embedding
        output_embedding = self.output_embedding
        cell = self.cell
        output_w = self.output_w
        output_b = self.output_b

        # Create Decoding Mask
        n = tf.shape(source)[0]
        s_l = tf.shape(source)[1]
        t_l = tf.shape(target)[1]
        mask = tf.to_float(tf.not_equal(target, 2))

        # Embedding Look Up
        source = tf.reshape(source, [-1])
        target = tf.reshape(target, [-1])
        i_embed = tf.nn.embedding_lookup(input_embedding, source)
        o_embed = tf.nn.embedding_lookup(output_embedding, target)
        i_embed = tf.reshape(i_embed, [n, s_l, self.en_embed_dim])
        o_embed = tf.reshape(o_embed, [n, t_l, self.de_embed_dim])

        # RNN
        h = tf.zeros(shape=[n, self.hid_dim])
        loss = 0.0
        for i in range(max_l):
            # Single step RNN calculate
            x = i_embed[:, i]
            h = cell(x, h)
            o = tf.concat(1, [x, h])
            o = tf.matmul(o, output_w) + output_b

            # Calculate score
            y = o_embed[:, i]
            candidate_score = tf.matmul(o, output_embedding, transpose_b=True)
            true_score = tf.reduce_sum(y * o, axis=-1)
            max_clip = tf.reduce_max(candidate_score, axis=-1)
            max_clip = tf.stop_gradient(max_clip)
            candidate_score = candidate_score - tf.reshape(max_clip, shape=[n, 1])
            true_score = true_score - max_clip
            # loss
            m = mask[:, i]
            step_loss = true_score - tf.log(tf.reduce_sum(tf.exp(candidate_score), axis=-1))
            loss += (step_loss * m)

        # average loss per data point
        loss = -tf.reduce_mean(loss / tf.reduce_sum(mask, axis=-1))

        gradient_update = tf.train.AdagradOptimizer(0.01).minimize(loss)
        return gradient_update, loss

    def build_prediction_graph(self, n_samples, max_l):
        # Get model parameters
        input_embedding = self.input_embedding
        output_embedding = self.output_embedding
        cell = self.cell
        output_w = self.output_w
        output_b = self.output_b

        # Create Decoding Mask

        # Embedding Look Up
        start = tf.zeros(shape=[n_samples, ], dtype="int32")
        x = tf.nn.embedding_lookup(input_embedding, start)

        # RNN
        h = tf.zeros(shape=[n_samples, self.hid_dim])
        prediction = []
        for i in range(max_l):
            # Single step RNN calculate
            h = cell(x, h)
            o = tf.concat(1, [x, h])
            o = tf.matmul(o, output_w) + output_b

            # Calculate score
            candidate_score = tf.matmul(o, output_embedding, transpose_b=True)
            max_clip = tf.reduce_max(candidate_score, axis=-1)
            max_clip = tf.stop_gradient(max_clip)
            candidate_score = candidate_score - tf.reshape(max_clip, shape=[n_samples, 1])
            candidate_score = tf.exp(candidate_score)
            candidate_prob = candidate_score / tf.reshape(tf.reduce_sum(candidate_score, axis=-1), [n_samples, 1])
            cdf = tf.cumsum(candidate_prob, axis=-1)
            threshold = tf.random_uniform([n_samples, 1], minval=0.0, maxval=1.0)
            cdf -= threshold
            samples = tf.cast(tf.greater_equal(cdf, 0), "float32")
            samples = tf.argmax(samples, axis=-1)
            x = tf.nn.embedding_lookup(input_embedding, samples)
            prediction.append(samples)

        return prediction


class BOWRNNBuilder(object):
    def __init__(self, vocab_size, en_embed_dim, de_embed_dim, hid_dim, latent_dim, max_l):
        self.vocab_size = vocab_size
        self.en_embed_dim = en_embed_dim
        self.de_embed_dim = de_embed_dim
        self.hid_dim = hid_dim
        self.max_len = max_l

        # Init word embedding
        self.input_embedding = tf.Variable(tf.random_uniform([vocab_size, de_embed_dim], minval=-0.05, maxval=0.05))
        self.output_embedding = tf.Variable(tf.random_uniform([vocab_size, de_embed_dim], minval=-0.05, maxval=0.05))

        # Init RNN cell
        self.cell = GRUCell(en_embed_dim, hid_dim)

        # Init the Auto encoder
        self.encode1_w = tf.Variable(tf.random_uniform([en_embed_dim*max_l, hid_dim], minval=-0.05, maxval=0.05))
        self.encode1_b = tf.Variable(tf.random_uniform([hid_dim, ], minval=-0.05, maxval=0.05))
        self.encode2_w = tf.Variable(tf.random_uniform([hid_dim, latent_dim], minval=-0.05, maxval=0.05))
        self.encode2_b = tf.Variable(tf.random_uniform([latent_dim, ], minval=-0.05, maxval=0.05))
        self.decode_w = tf.Variable(tf.random_uniform([latent_dim, de_embed_dim], minval=-0.05, maxval=0.05))
        self.decode_b = tf.Variable(tf.random_uniform([de_embed_dim, ], minval=-0.05, maxval=0.05))

        # Output projection layer
        self.output_w = tf.Variable(tf.random_uniform([en_embed_dim + hid_dim + latent_dim, de_embed_dim], minval=-0.05, maxval=0.05))
        self.output_b = tf.Variable(tf.random_uniform([de_embed_dim, ], minval=-0.05, maxval=0.05))

    def build_training_graph(self):
        source = tf.placeholder(dtype=tf.int32, name="source", shape=[None, self.max_len])
        target = tf.placeholder(dtype=tf.int32, name="target", shape=[None, self.max_len])
        alpha = tf.placeholder(dtype=tf.float32, name="alpha")

        # Get model parameters
        input_embedding = self.input_embedding
        output_embedding = self.output_embedding
        cell = self.cell
        output_w = self.output_w
        output_b = self.output_b

        encode1_w = self.encode1_w
        encode1_b = self.encode1_b
        encode2_w = self.encode2_w
        encode2_b = self.encode2_b
        decode_w = self.decode_w
        decode_b = self.decode_b

        # Create Decoding Mask
        n = tf.shape(source)[0]
        s_l = tf.shape(source)[1]
        t_l = tf.shape(target)[1]
        mask = tf.to_float(tf.not_equal(target, 8195))
        e_mask = tf.to_float(tf.not_equal(source, 8195))

        # Embedding Look Up
        source = tf.reshape(source, [-1])
        target = tf.reshape(target, [-1])
        i_embed = tf.nn.embedding_lookup(input_embedding, source)
        o_embed = tf.nn.embedding_lookup(output_embedding, target)
        i_embed = tf.reshape(i_embed, [n, s_l, self.en_embed_dim])
        i_embed = i_embed * tf.reshape(e_mask, [n, s_l, 1])
        o_embed = tf.reshape(o_embed, [n, t_l, self.de_embed_dim])

        # Auto Encoder

        ae_input = tf.reshape(i_embed, [n, self.en_embed_dim * self.max_len])
        encode1 = tf.nn.relu(tf.matmul(ae_input, encode1_w) + encode1_b)
        z = tf.nn.relu(tf.matmul(encode1, encode2_w) + encode2_b)
        decode = tf.nn.relu(tf.matmul(z, decode_w) + decode_b)

        # Auto Encoder Loss
        candidate_score = tf.matmul(decode, output_embedding, transpose_b=True)
        max_clip = tf.stop_gradient(tf.reduce_max(candidate_score, axis=-1))
        candidate_score = candidate_score - tf.reshape(max_clip, [n, 1])

        d_h = tf.reshape(decode, [n, 1, self.de_embed_dim])
        true_output = tf.nn.embedding_lookup(output_embedding, source)
        true_output = tf.reshape(true_output, [n, t_l, self.de_embed_dim])
        target_score = tf.reduce_sum(d_h * true_output, axis=-1)
        target_score = target_score - tf.reshape(max_clip, [n, 1])
        candidate_score = tf.exp(candidate_score)
        ae_loss = target_score - tf.log(tf.reshape(tf.reduce_sum(candidate_score, axis=-1), [n, 1]))
        ae_loss = - tf.reduce_mean(ae_loss*mask)
        # RNN
        h = tf.zeros(shape=[n, self.hid_dim])
        rnn_loss = 0.0
        for i in range(self.max_len):
            # Single step RNN calculate
            x = i_embed[:, i]
            h = cell(x, h)
            o = tf.concat(1, [x, z, h])
            o = tf.matmul(o, output_w) + output_b

            # Calculate score
            y = o_embed[:, i]
            candidate_score = tf.matmul(o, output_embedding, transpose_b=True)
            true_score = tf.reduce_sum(y * o, axis=-1)
            max_clip = tf.reduce_max(candidate_score, axis=-1)
            max_clip = tf.stop_gradient(max_clip)
            candidate_score = candidate_score - tf.reshape(max_clip, shape=[n, 1])
            true_score = true_score - max_clip
            # loss
            m = mask[:, i]
            step_loss = true_score - tf.log(tf.reduce_sum(tf.exp(candidate_score), axis=-1))
            rnn_loss += (step_loss * m)

        # average loss per data point
        rnn_loss = -tf.reduce_mean(rnn_loss / tf.reduce_sum(mask, axis=-1))
        loss = (1.0 - alpha)*ae_loss + alpha*rnn_loss
        gradient_update = tf.train.AdagradOptimizer(0.01).minimize(loss)
        return gradient_update, rnn_loss, ae_loss

    def build_prediction_graph(self, n_samples, max_l):
        source = tf.placeholder(dtype=tf.int32, name="source", shape=[None, self.max_len])
        # Get model parameters
        input_embedding = self.input_embedding
        output_embedding = self.output_embedding
        cell = self.cell
        output_w = self.output_w
        output_b = self.output_b

        encode1_w = self.encode1_w
        encode1_b = self.encode1_b
        encode2_w = self.encode2_w
        encode2_b = self.encode2_b
        decode_w = self.decode_w
        decode_b = self.decode_b

        # Create Decoding Mask
        n = tf.shape(source)[0]
        s_l = tf.shape(source)[1]
        # Embedding Look Up
        source = tf.reshape(source, [-1])
        i_embed = tf.nn.embedding_lookup(input_embedding, source)
        i_embed = tf.reshape(i_embed, [n, s_l, self.en_embed_dim])

        k_num = 100
        # Embedding Look Up
        start = tf.zeros(shape=[n_samples*k_num, ], dtype="int32")
        x = tf.nn.embedding_lookup(input_embedding, start)

        ae_input = tf.reshape(i_embed, [n, self.en_embed_dim * self.max_len])
        encode1 = tf.nn.relu(tf.matmul(ae_input, encode1_w) + encode1_b)
        z = tf.nn.relu(tf.matmul(encode1, encode2_w) + encode2_b)
        decode = tf.nn.relu(tf.matmul(z, decode_w) + decode_b)
        candidate_score = tf.matmul(decode, output_embedding, transpose_b=True)
        candidate_score = tf.exp(candidate_score)
        candidate_score = candidate_score / tf.reduce_sum(candidate_score, axis=-1, keep_dims=True)
        auto_out, auto_indices = tf.nn.top_k(candidate_score, k=8195)

        # RNN
        h = tf.zeros(shape=[n_samples*k_num, self.hid_dim])
        max_z = z
        max_h = tf.zeros(shape=[n_samples, self.hid_dim])
        max_start = tf.zeros(shape=[n_samples, ], dtype="int32")
        max_x = tf.nn.embedding_lookup(input_embedding, max_start)
        prediction = []
        max_pred = []
        rnn_z = tf.tile(z, [k_num, 1])
        score = tf.zeros(shape=[n_samples*k_num])
        for i in range(max_l):
            # Single step RNN calculate
            h = cell(x, h)
            max_h = cell(max_x, max_h)
            max_o = tf.concat(1, [max_x, max_z, max_h])
            o = tf.concat(1, [x, rnn_z, h])
            o = tf.matmul(o, output_w) + output_b
            max_o = tf.matmul(max_o, output_w) + output_b

            # Calculate score
            candidate_score = tf.matmul(o, output_embedding, transpose_b=True)
            max_cadidate_score = tf.matmul(max_o, output_embedding, transpose_b=True)
            max_sample = tf.argmax(max_cadidate_score, axis=1)
            max_clip = tf.reduce_max(candidate_score, axis=-1)
            max_clip = tf.stop_gradient(max_clip)
            candidate_score = candidate_score - tf.reshape(max_clip, shape=[n_samples*k_num, 1])
            candidate_score = tf.exp(candidate_score)
            candidate_prob = candidate_score / tf.reshape(tf.reduce_sum(candidate_score, axis=-1), [n_samples*k_num, 1])
            dist = Categorical(p=candidate_prob)
            samples = dist.sample()
            x = tf.nn.embedding_lookup(input_embedding, samples)
            score += tf.reduce_sum(o*x, axis=-1)
            max_x = tf.nn.embedding_lookup(input_embedding, max_sample)
            prediction.append(tf.reshape(samples, [n_samples, k_num]))
            max_pred.append(max_sample)

        score = tf.reshape(score, [n_samples, k_num])
        max_sequence = tf.argmax(score, axis=1)

        return prediction, auto_out, auto_indices, max_pred, max_sequence, z


class AERNNBuilder(object):
    def __init__(self, vocab_size, en_embed_dim, de_embed_dim, hid_dim, latent_dim, max_l):
        self.vocab_size = vocab_size
        self.en_embed_dim = en_embed_dim
        self.de_embed_dim = de_embed_dim
        self.hid_dim = hid_dim
        self.max_len = max_l

        # Init word embedding
        self.input_embedding = tf.Variable(tf.random_uniform([vocab_size, de_embed_dim], minval=-0.05, maxval=0.05))
        self.output_embedding = tf.Variable(tf.random_uniform([vocab_size, de_embed_dim], minval=-0.05, maxval=0.05))

        # Init RNN cell
        self.cell = GRUCell(en_embed_dim, hid_dim)

        # Init the Auto encoder
        self.encode1_w = tf.Variable(tf.random_uniform([en_embed_dim*max_l, hid_dim], minval=-0.05, maxval=0.05))
        self.encode1_b = tf.Variable(tf.random_uniform([hid_dim, ], minval=-0.05, maxval=0.05))
        self.encode2_w = tf.Variable(tf.random_uniform([hid_dim, latent_dim], minval=-0.05, maxval=0.05))
        self.encode2_b = tf.Variable(tf.random_uniform([latent_dim, ], minval=-0.05, maxval=0.05))
        self.decode1_w = tf.Variable(tf.random_uniform([latent_dim, de_embed_dim], minval=-0.05, maxval=0.05))
        self.decode1_b = tf.Variable(tf.random_uniform([de_embed_dim, ], minval=-0.05, maxval=0.05))
        self.decode2_w = tf.Variable(tf.random_uniform([de_embed_dim, de_embed_dim*max_l], minval=-0.05, maxval=0.05))
        self.decode2_b = tf.Variable(tf.random_uniform([de_embed_dim*max_l, ], minval=-0.05, maxval=0.05))

        # Output projection layer
        self.output_w = tf.Variable(tf.random_uniform([en_embed_dim + hid_dim + latent_dim, de_embed_dim], minval=-0.05, maxval=0.05))
        self.output_b = tf.Variable(tf.random_uniform([de_embed_dim, ], minval=-0.05, maxval=0.05))

    def build_training_graph(self):
        source = tf.placeholder(dtype=tf.int32, name="source", shape=[None, self.max_len])
        target = tf.placeholder(dtype=tf.int32, name="target", shape=[None, self.max_len])
        alpha = tf.placeholder(dtype=tf.float32, name="alpha")

        # Get model parameters
        input_embedding = self.input_embedding
        output_embedding = self.output_embedding
        cell = self.cell
        output_w = self.output_w
        output_b = self.output_b

        encode1_w = self.encode1_w
        encode1_b = self.encode1_b
        encode2_w = self.encode2_w
        encode2_b = self.encode2_b
        decode1_w = self.decode1_w
        decode1_b = self.decode1_b
        decode2_w = self.decode2_w
        decode2_b = self.decode2_b

        # Create Decoding Mask
        n = tf.shape(source)[0]
        s_l = tf.shape(source)[1]
        t_l = tf.shape(target)[1]
        mask = tf.to_float(tf.not_equal(target, 8195))
        e_mask = tf.to_float(tf.not_equal(source, 8195))

        # Embedding Look Up
        source = tf.reshape(source, [-1])
        target = tf.reshape(target, [-1])
        i_embed = tf.nn.embedding_lookup(input_embedding, source)
        o_embed = tf.nn.embedding_lookup(output_embedding, target)
        i_embed = tf.reshape(i_embed, [n, s_l, self.en_embed_dim])
        i_embed = i_embed * tf.reshape(e_mask, [n, s_l, 1])
        o_embed = tf.reshape(o_embed, [n, t_l, self.de_embed_dim])

        # Auto Encoder

        ae_input = tf.reshape(i_embed, [n, self.en_embed_dim * self.max_len])
        encode1 = tf.nn.relu(tf.matmul(ae_input, encode1_w) + encode1_b)
        z = tf.nn.relu(tf.matmul(encode1, encode2_w) + encode2_b)
        decode = tf.nn.relu(tf.matmul(z, decode1_w) + decode1_b)
        decode = tf.nn.relu(tf.matmul(decode, decode2_w) + decode2_b)
        decode = tf.reshape(decode, [n, t_l, self.de_embed_dim])
        decode = tf.reshape(decode, [n*t_l, self.de_embed_dim])
        # Auto Encoder Loss
        candidate_score = tf.matmul(decode, output_embedding, transpose_b=True)
        max_clip = tf.stop_gradient(tf.reduce_max(candidate_score, axis=-1))
        candidate_score = candidate_score - tf.reshape(max_clip, [n*t_l, 1])
        true_output = tf.nn.embedding_lookup(output_embedding, source)
        target_score = tf.reduce_sum(decode * true_output, axis=-1)
        target_score = target_score - max_clip
        candidate_score = tf.exp(candidate_score)
        ae_loss = target_score - tf.log(tf.reduce_sum(candidate_score, axis=-1))
        ae_loss = tf.reshape(ae_loss, [n, t_l])
        ae_loss = - tf.reduce_mean(ae_loss*mask)
        # RNN
        h = tf.zeros(shape=[n, self.hid_dim])
        rnn_loss = 0.0
        for i in range(self.max_len):
            # Single step RNN calculate
            x = i_embed[:, i]
            h = cell(x, h)
            o = tf.concat(1, [x, z, h])
            o = tf.matmul(o, output_w) + output_b

            # Calculate score
            y = o_embed[:, i]
            candidate_score = tf.matmul(o, output_embedding, transpose_b=True)
            true_score = tf.reduce_sum(y * o, axis=-1)
            max_clip = tf.reduce_max(candidate_score, axis=-1)
            max_clip = tf.stop_gradient(max_clip)
            candidate_score = candidate_score - tf.reshape(max_clip, shape=[n, 1])
            true_score = true_score - max_clip
            # loss
            m = mask[:, i]
            step_loss = true_score - tf.log(tf.reduce_sum(tf.exp(candidate_score), axis=-1))
            rnn_loss += (step_loss * m)

        # average loss per data point
        rnn_loss = -tf.reduce_mean(rnn_loss / tf.reduce_sum(mask, axis=-1))
        loss = (1.0 - alpha)*ae_loss + alpha*rnn_loss
        gradient_update = tf.train.AdagradOptimizer(0.01).minimize(loss)
        return gradient_update, rnn_loss, ae_loss

    def build_prediction_graph(self, n_samples, max_l):
        source = tf.placeholder(dtype=tf.int32, name="source", shape=[None, self.max_len])
        # Get model parameters
        input_embedding = self.input_embedding
        output_embedding = self.output_embedding
        cell = self.cell
        output_w = self.output_w
        output_b = self.output_b

        encode1_w = self.encode1_w
        encode1_b = self.encode1_b
        encode2_w = self.encode2_w
        encode2_b = self.encode2_b
        decode1_w = self.decode1_w
        decode1_b = self.decode1_b
        decode2_w = self.decode2_w
        decode2_b = self.decode2_b

        n = tf.shape(source)[0]
        s_l = tf.shape(source)[1]
        e_mask = tf.to_float(tf.not_equal(source, 8195))

        # Embedding Look Up
        source = tf.reshape(source, [-1])
        i_embed = tf.nn.embedding_lookup(input_embedding, source)
        i_embed = tf.reshape(i_embed, [n, s_l, self.en_embed_dim])
        i_embed = i_embed * tf.reshape(e_mask, [n, s_l, 1])

        # Auto Encoder

        ae_input = tf.reshape(i_embed, [n, self.en_embed_dim * self.max_len])
        encode1 = tf.nn.relu(tf.matmul(ae_input, encode1_w) + encode1_b)
        z = tf.nn.relu(tf.matmul(encode1, encode2_w) + encode2_b)
        decode = tf.nn.relu(tf.matmul(z, decode1_w) + decode1_b)
        decode = tf.nn.relu(tf.matmul(decode, decode2_w) + decode2_b)
        decode = tf.reshape(decode, [n, self.max_len, self.de_embed_dim])
        decode = tf.reshape(decode, [n * self.max_len, self.de_embed_dim])
        # Auto Encoder Loss
        candidate_score = tf.matmul(decode, output_embedding, transpose_b=True)
        max_clip = tf.stop_gradient(tf.reduce_max(candidate_score, axis=-1))
        candidate_score = candidate_score - tf.reshape(max_clip, [n * self.max_len, 1])
        candidate_score = tf.exp(candidate_score)
        candidate_probs = candidate_score / tf.reduce_sum(candidate_score, axis=-1, keep_dims=True)
        dist = Categorical(p=candidate_probs)
        ae_sample = dist.sample(sample_shape=(5, ))
        ae_sample = tf.reshape(ae_sample, [5, n, self.max_len])
        # RNN
        start = tf.zeros(shape=[n_samples * 5, ], dtype="int32")
        x = tf.nn.embedding_lookup(input_embedding, start)
        h = tf.zeros(shape=[n_samples*5, self.hid_dim])
        max_z = z
        max_h = tf.zeros(shape=[n_samples, self.hid_dim])
        max_start = tf.zeros(shape=[n_samples, ], dtype="int32")
        max_x = tf.nn.embedding_lookup(input_embedding, max_start)
        prediction = []
        max_pred = []
        rnn_z = tf.tile(z, [5, 1])
        for i in range(max_l):
            # Single step RNN calculate
            h = cell(x, h)
            max_h = cell(max_x, max_h)
            max_o = tf.concat(1, [max_x, max_z, max_h])
            o = tf.concat(1, [x, rnn_z, h])
            o = tf.matmul(o, output_w) + output_b
            max_o = tf.matmul(max_o, output_w) + output_b

            # Calculate score
            candidate_score = tf.matmul(o, output_embedding, transpose_b=True)
            max_cadidate_score = tf.matmul(max_o, output_embedding, transpose_b=True)
            max_sample = tf.argmax(max_cadidate_score, axis=1)
            max_clip = tf.reduce_max(candidate_score, axis=-1)
            max_clip = tf.stop_gradient(max_clip)
            candidate_score = candidate_score - tf.reshape(max_clip, shape=[n_samples*5, 1])
            candidate_score = tf.exp(candidate_score)
            candidate_prob = candidate_score / tf.reshape(tf.reduce_sum(candidate_score, axis=-1), [n_samples*5, 1])
            dist = Categorical(p=candidate_prob)
            samples = dist.sample()
            x = tf.nn.embedding_lookup(input_embedding, samples)
            max_x = tf.nn.embedding_lookup(input_embedding, max_sample)
            prediction.append(tf.reshape(samples, [n_samples, 5]))
            max_pred.append(max_sample)

        return prediction, ae_sample, max_pred


class AEBuilder(object):
    def __init__(self, vocab_size, en_embed_dim, de_embed_dim, hid_dim, latent_dim, max_l):
        self.vocab_size = vocab_size
        self.en_embed_dim = en_embed_dim
        self.de_embed_dim = de_embed_dim
        self.hid_dim = hid_dim
        self.max_len = max_l

        # Init word embedding
        self.input_embedding = tf.Variable(tf.random_uniform([vocab_size, de_embed_dim], minval=-0.05, maxval=0.05))
        self.output_embedding = tf.Variable(tf.random_uniform([vocab_size, de_embed_dim], minval=-0.05, maxval=0.05))

        # Init the Auto encoder
        self.encode1_w = tf.Variable(tf.random_uniform([en_embed_dim*max_l, hid_dim], minval=-0.05, maxval=0.05))
        self.encode1_b = tf.Variable(tf.random_uniform([hid_dim, ], minval=-0.05, maxval=0.05))
        self.encode2_w = tf.Variable(tf.random_uniform([hid_dim, latent_dim], minval=-0.05, maxval=0.05))
        self.encode2_b = tf.Variable(tf.random_uniform([latent_dim, ], minval=-0.05, maxval=0.05))
        self.decode_w = tf.Variable(tf.random_uniform([latent_dim, de_embed_dim], minval=-0.05, maxval=0.05))
        self.decode_b = tf.Variable(tf.random_uniform([de_embed_dim, ], minval=-0.05, maxval=0.05))

    def build_training_graph(self):
        source = tf.placeholder(dtype=tf.int32, name="source", shape=[None, self.max_len])

        # Get model parameters
        input_embedding = self.input_embedding
        output_embedding = self.output_embedding

        encode1_w = self.encode1_w
        encode1_b = self.encode1_b
        encode2_w = self.encode2_w
        encode2_b = self.encode2_b
        decode_w = self.decode_w
        decode_b = self.decode_b

        # Create Decoding Mask
        n = tf.shape(source)[0]
        s_l = tf.shape(source)[1]
        e_mask = tf.to_float(tf.not_equal(source, 8195))

        # Embedding Look Up
        source = tf.reshape(source, [-1])
        i_embed = tf.nn.embedding_lookup(input_embedding, source)
        i_embed = tf.reshape(i_embed, [n, s_l, self.en_embed_dim])
        i_embed = i_embed * tf.reshape(e_mask, [n, s_l, 1])

        # Auto Encoder

        ae_input = tf.reshape(i_embed, [n, self.en_embed_dim * self.max_len])
        encode1 = tf.nn.relu(tf.matmul(ae_input, encode1_w) + encode1_b)
        z = tf.nn.relu(tf.matmul(encode1, encode2_w) + encode2_b)
        decode = tf.nn.relu(tf.matmul(z, decode_w) + decode_b)

        # Auto Encoder Loss
        candidate_score = tf.matmul(decode, output_embedding, transpose_b=True)
        max_clip = tf.stop_gradient(tf.reduce_max(candidate_score, axis=-1))
        candidate_score = candidate_score - tf.reshape(max_clip, [n, 1])

        d_h = tf.reshape(decode, [n, 1, self.de_embed_dim])
        true_output = tf.nn.embedding_lookup(output_embedding, source)
        true_output = tf.reshape(true_output, [n, s_l, self.de_embed_dim])
        target_score = tf.reduce_sum(d_h * true_output, axis=-1)
        target_score = target_score - tf.reshape(max_clip, [n, 1])
        candidate_score = tf.exp(candidate_score)
        ae_loss = target_score - tf.log(tf.reshape(tf.reduce_sum(candidate_score, axis=-1), [n, 1]))
        ae_loss = - tf.reduce_mean(ae_loss*e_mask)

        # average loss per data point
        gradient_update = tf.train.AdagradOptimizer(0.01).minimize(ae_loss)
        return gradient_update, ae_loss

    def build_prediction_graph(self, n_samples, max_l):
        source = tf.placeholder(dtype=tf.int32, name="source", shape=[None, self.max_len])
        # Get model parameters
        input_embedding = self.input_embedding
        output_embedding = self.output_embedding

        encode1_w = self.encode1_w
        encode1_b = self.encode1_b
        encode2_w = self.encode2_w
        encode2_b = self.encode2_b
        decode_w = self.decode_w
        decode_b = self.decode_b

        # Create Decoding Mask
        n = tf.shape(source)[0]
        s_l = tf.shape(source)[1]
        # Embedding Look Up
        source = tf.reshape(source, [-1])
        i_embed = tf.nn.embedding_lookup(input_embedding, source)
        i_embed = tf.reshape(i_embed, [n, s_l, self.en_embed_dim])

        # Embedding Look Up
        start = tf.zeros(shape=[n_samples, ], dtype="int32")
        x = tf.nn.embedding_lookup(input_embedding, start)

        ae_input = tf.reshape(i_embed, [n, self.en_embed_dim * self.max_len])
        encode1 = tf.nn.relu(tf.matmul(ae_input, encode1_w) + encode1_b)
        z = tf.nn.relu(tf.matmul(encode1, encode2_w) + encode2_b)
        decode = tf.nn.relu(tf.matmul(z, decode_w) + decode_b)
        candidate_score = tf.matmul(decode, output_embedding, transpose_b=True)
        candidate_score = tf.exp(candidate_score)
        candidate_score = candidate_score / tf.reduce_sum(candidate_score, axis=-1, keep_dims=True)
        auto_out, auto_indices = tf.nn.top_k(candidate_score, k=8195)

        return auto_out, auto_indices


def debug_test(device):
    vocab = []
    vocab_idx = {}
    idx = 0
    test_ae = False
    test_rnn = True
    with open("Data/sentence_20k/vocab.txt", "r") as v:
        for line in v:
            vocab.append(line.rstrip("\n"))
            vocab_idx[line.rstrip("\n")] = idx
            idx += 1

    with tf.device(device):
        with open("Data/idx/multi_idx.txt", "r") as data:
            data = json.loads(data.read())
            subset = sorted(data, key=lambda d: len(d))
            l = len(subset[-1])
            builder = BOWRNNBuilder(8196, 256, 256, 512, 512, 16)
            prediction_graph = builder.build_prediction_graph(len(subset), 16)
            init = tf.global_variables_initializer()
            sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
            sess.run(init)
            saver = tf.train.Saver()
            saver.restore(sess, "code_outputs/2017_08_03_09_40_06/final_model_params.ckpt")
            source = None
            start = time.clock()
            for datapoint in subset:
                s = np.array(datapoint)
                if len(s) != l:
                    # pad sentences
                    s = np.append(s, [8195] * (l - len(s)))
                if source is None:
                    source = s.reshape((1, s.shape[0]))
                else:
                    source = np.concatenate([source, s.reshape((1, s.shape[0]))])
            model_input = {"source:0": source[:, :-1]}
            rnn_pred, auto_out, ae_pred, max_pred, max_sequence, z = sess.run(prediction_graph, feed_dict=model_input)
            np.save("Data/Samples/BOW/multi.npy", z)
            #l = len(rnn_indices)
        for n in range(len(subset)):
            s_sentence = ""
            for idx in subset[n]:
                if idx == 0:
                    continue
                token = vocab[idx]
                s_sentence = s_sentence + " " + token
            print("Origin : " + s_sentence)
            max_prediction = ""
            for t in range(16):
                m_p = max_pred[t]
                m_p = m_p[n]
                max_prediction += (" " + vocab[m_p])
            print("Greedy : " + max_prediction)
            for i in range(5):
                rnn_sample = ""
                ae_sample = ""
                for t in range(16):
                    r_p = rnn_pred[t]
                    r_p = r_p[n]
                    r_p = r_p[i]
                    if r_p == 1:
                        break
                    rnn_sample += (" " + vocab[r_p])
                print(str(i+1) + "th RNN Sample : " + rnn_sample)
                if test_ae:
                    for t in range(15):
                        a_s = ae_pred[i]
                        a_s = a_s[n]
                        a_s = a_s[t+1]
                        if a_s == 1:
                            break
                        ae_sample += (" " + vocab[a_s])
                    print(str(i + 1) + "th AE Sample : " + ae_sample)
            max_like_idx = max_sequence[n]
            max_like_sen = ""
            for t in range(16):
                r_p = rnn_pred[t]
                r_p = r_p[n]
                r_p = r_p[max_like_idx]
                max_like_sen += (" " + vocab[r_p])
            print("max like : " + max_like_sen)


def training(device, out_dir):
    print(" AERNN 0.5 ")
    with tf.device(device):
        builder = BOWRNNBuilder(8196, 256, 256, 512, 512, 16)
        gradient_update, rnn_loss, ae_loss = builder.build_training_graph()
        init = tf.global_variables_initializer()
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        sess.run(init)
        saver = tf.train.Saver()
        #saver.restore(sess, "code_outputs/2017_07_20_14_28_25/final_model_params.ckpt")
    # Load training set
        with open("Data/sentence_20k/idx.txt", "r") as data:
            data = json.loads(data.read())

        r_loss = []
        a_loss = []
        epoch_size = len(data)
        print("training for " + str(int(epoch_size/100)) + " iterations")
        for iters in range(10000):
            # get 300 random data points from training data
            batch_indices = np.random.choice(len(data), 1000, replace=False)
            mini_batch = [data[ind] for ind in batch_indices]
            # sort the 300 data points by length
            mini_batch = sorted(mini_batch, key=lambda d: len(d))
            mini_batch = np.array(mini_batch)
            # divide 300 data points into 10 mini batch
            mini_batchs = np.split(mini_batch, 10)

            for m in mini_batchs:
                # get the maximum length in mini batch
                l = len(m[-1])
                source_l = []
                source = None
                for datapoint in m:
                    s = np.array(datapoint)
                    source_l.append(len(s))
                    if len(s) != l:
                        # pad sentences
                        s = np.append(s, [8195] * (l - len(s)))

                    if source is None:
                        source = s.reshape((1, s.shape[0]))
                    else:
                        source = np.concatenate([source, s.reshape((1, s.shape[0]))])
                model_input = {"source:0": source[:, :-1], "target:0": source[:, 1:], "alpha:0": 0.5}
                start = time.clock()
                _, r_l, a_l = sess.run([gradient_update, rnn_loss, ae_loss], feed_dict=model_input)
                a_loss.append(a_l)
                r_loss.append(r_l)
                if iters % 40 == 0:
                    print("The training time per mini batch is " + str(time.clock()-start))
                    print("The maximum sentence length in mini-batch is " + str(l))
                    print("The rnn loss is : " + str(r_l))
                    print("The ae loss is : " + str(a_l))
                    print("")

            if iters % 2000 == 0 and iters is not 0:
                np.save(os.path.join(out_dir, 'rnn_loss.npy'), r_loss)
                np.save(os.path.join(out_dir, 'ae_loss.npy'), a_loss)
                saver.save(sess, out_dir+"/model_params.ckpt")

        np.save(os.path.join(out_dir, 'rnn_loss.npy'), r_loss)
        np.save(os.path.join(out_dir, 'ae_loss.npy'), a_loss)
        saver.save(sess, out_dir + "/final_model_params.ckpt")
