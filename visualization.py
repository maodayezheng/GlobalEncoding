from tensorflow.contrib.tensorboard.plugins import projector
import tensorflow as tf
import os
import numpy as np

N = 10000
D = 200

multi = np.load()
quotes = np.load()
random = np.load()
straight = np.load()
e = np.concatenate([multi, quotes. random, straight], axis=0)

embedding_var = tf.Variable(e, name='word_embedding')
config = projector.ProjectorConfig()

# You can add multiple embeddings. Here we add only one.
embedding = config.embeddings.add()
embedding.tensor_name = embedding_var.name
# Link this tensor to its metadata file (e.g. labels).
embedding.metadata_path = os.path.join(LOG_DIR, 'metadata.tsv')

# Use the same LOG_DIR where you stored your checkpoint.
summary_writer = tf.summary.FileWriter(LOG_DIR)

# The next line writes a projector_config.pbtxt in the LOG_DIR. TensorBoard will
# read this file during startup.

projector.visualize_embeddings(summary_writer, config)