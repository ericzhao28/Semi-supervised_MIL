import tensorflow as tf


def EncoderDecoder(X, batch_s, init, hidden, cell, t_pred):
  '''
  Seq2Seq encoder-decoder module.
  '''

  with tf.variable_scope("encoderdecoder") as scope:
    # Reshape X to [batch, t_in, all]
    X = tf.reshape(X, (X.shape[0], X.shape[1], -1))
    # Reshape X to [batch, all], [batch, all].. t_in.
    X = tf.split(X, X.shape[1], axis=1)

    # Build LSTM: do not use relu :(
    lstm = cell(num_units=hidden, initializer=init)
    init_state = lstm.zero_state(batch_s, "float32")

    # Encoder
    output, state = lstm(X[0], init_state)
    for datum in X[1:]:
      scope.reuse_variables()
      output, state = lstm(datum, state)

    # Decoder
    x_dummy = tf.zeros_like(X[0], "float32")
    Y = []
    for t in range(t_pred):
      scope.reuse_variables()
      output, state = lstm(x_dummy, state)
      Y.append(output)

    # Return the inferences..
    return tf.reshape(tf.concat(Y, axis=1), [batch_s * t_pred, -1])


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
  with tf.variable_scope("fc"):
    W = tf.Variable(init([X.shape[-1], output]))
    b = tf.Variable(init((output,)))
    Y = tf.matmul(X, W) + b
    return activ(Y)

