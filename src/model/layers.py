import tensorflow as tf


def Conv2d(X, batch_s, channels, strides, height, width, init, activ):
  """
  Conv2d layer.
  x should be in shape of: [batch, height, width, channels]
  W should be in shape of: [filter_height, filter_width,
                            in channel, out channels]
  b is a scalar bias.
  """

  with tf.variable_scope("conv2d"):
    W = tf.Variable(init((height, width, X.shape[-1], channels)))
    b = tf.Variable(init((channels,)))

    XW = tf.nn.conv2d(X, W, strides=[1, strides, strides, 1], padding='SAME')
    Y = tf.nn.bias_add(XW, b)

    return activ(Y)


def FC(X, output, init, activ):
  """
  Build fully-connected model.
  """
  with tf.variable_scope("fc"):
    W = tf.Variable(init([X.shape[-1], output]))
    b = tf.Variable(init((output,)))
    Y = tf.matmul(X, W) + b
    return activ(Y)


