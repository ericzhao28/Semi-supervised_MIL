import tensorflow as tf


def conv2d(X, W, b, strides=1):
  """
  x should be in shape of: [batch, height, width, channels]
  W should be in shape of: [filter_height, filter_width,
                            in channel, out channels]
  b is a scalar bias.
  """

  X = tf.nn.conv2d(X, W, strides=[1, strides, strides, 1], padding='SAME')
  X = tf.nn.bias_add(X, b)
  return tf.nn.relu(X)


def fc(X, W, b):
  X = tf.matmul(X, W) + b
  return tf.nn.relu(X)

