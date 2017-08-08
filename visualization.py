from tensorflow.contrib.tensorboard.plugins import projector
import tensorflow as tf
import os

N = 10000
D = 200
embedding_var = tf.Variable(tf.random_normal([N, D]), name='word_embedding')
source = tf.placeholder(dtype=tf.int32, name="source", shape=[None, 512])
# Format: tensorflow/tensorboard/plugins/projector/projector_config.proto
config = projector.ProjectorConfig()

# You can add multiple embeddings. Here we add only one.
embedding = config.embeddings.add()
embedding.tensor_name = source.name
# Link this tensor to its metadata file (e.g. labels).
embedding.metadata_path = os.path.join(LOG_DIR, 'metadata.tsv')

# Use the same LOG_DIR where you stored your checkpoint.
summary_writer = tf.summary.FileWriter(LOG_DIR)

# The next line writes a projector_config.pbtxt in the LOG_DIR. TensorBoard will
# read this file during startup.

projector.visualize_embeddings(summary_writer, config)