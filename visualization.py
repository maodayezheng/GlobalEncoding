from tensorflow.contrib.tensorboard.plugins import projector
import tensorflow as tf
import os
import numpy as np

N = 10000
D = 200

multi = np.load("visualization/ae_multi.npy")
print(multi.shape)
quotes = np.load("visualization/ae_quotes.npy")
print(quotes.shape)
random = np.load("visualization/ae_random.npy")
print(random.shape)
straight = np.load("visualization/ae_straight.npy")
print(straight.shape)
e = np.concatenate([multi, quotes, random, straight], axis=0)

embedding_var = tf.Variable(e, name='word_embedding')
init = tf.global_variables_initializer()
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
sess.run(init)
saver = tf.train.Saver()
saver.save(sess, "visualization/latent_embedding.ckpt")

config = projector.ProjectorConfig()

# You can add multiple embeddings. Here we add only one.
embedding = config.embeddings.add()
embedding.tensor_name = embedding_var.name
# Link this tensor to its metadata file (e.g. labels).
embedding.metadata_path = os.path.join("visualization", 'iris-latin1.tsv')

# Use the same LOG_DIR where you stored your checkpoint.
summary_writer = tf.summary.FileWriter("visualization")

# The next line writes a projector_config.pbtxt in the LOG_DIR. TensorBoard will
# read this file during startup.

projector.visualize_embeddings(summary_writer, config)
