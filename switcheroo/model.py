import tensorflow as tf
from tf_utils import conv2d, fc
from config import BATCH, T_in, T_pred, std_dev, IMG_CH, IMG_H, IMG_W


class LSTMAutoEncoder(object):
  def __init__(self):
    self.weights = {
        # wcx are conv2d filter weights of shape: [filter_h, filter_w,
        #                                          in_ch, out_ch]
        'wc1': tf.Variable(tf.random_normal([5, 5, IMG_CH, 24],
                                            stddev=std_dev)),
        'wc2': tf.Variable(tf.random_normal([5, 5, 24, 64], stddev=std_dev)),
        'wc3': tf.Variable(tf.random_normal([5, 5, 64, 64], stddev=std_dev)),
        'wfc1': tf.Variable(tf.random_normal([1024, IMG_H * IMG_W * IMG_CH],
                                             stddev=std_dev))
    }
    self.biases = {
        'bc1': tf.Variable(tf.random_normal([24], stddev=0)),
        'bc2': tf.Variable(tf.random_normal([64], stddev=0)),
        'bc3': tf.Variable(tf.random_normal([64], stddev=0)),
        'bfc1': tf.Variable(tf.random_normal([IMG_H * IMG_W * IMG_CH],
                                             stddev=0))
    }

  def EncoderDecoder(self, data):
    # Input shape: [BATCH, T_in, -1]
    # Output shape: [BATCH * T_pred, -1]
    with tf.variable_scope("LSTM") as scope:
      lstm = tf.contrib.rnn.LSTMCell(num_units=512)
      state = lstm.zero_state(BATCH, "float")
      datum = tf.split(data, T_in, axis=1)

      # Encoder
      for t in range(T_in):
        if t != 0:
         # Reuse scope for all subsequent timesteps.
          scope.reuse_variables()
        tmp = tf.reshape(datum[t], [BATCH, -1])
        output, state = lstm(tmp, state)

      # Decoder
      tmp = tf.reshape(datum[0], [BATCH, -1])
      zero_ = tf.zeros_like(tmp, "float")
      output_list = []
      for t in range(T_pred):
        scope.reuse_variables()
        output, state = lstm(zero_, state)
        output_list.append(output)

      # Return the inferences..
      out = tf.concat(output_list, axis=1)
      return tf.reshape(out, [BATCH * T_pred, -1])


def build_model(net):
  # Construct model..
  X = tf.placeholder("float", [BATCH, T_in, IMG_H, IMG_W, IMG_CH])
  Y = tf.placeholder("float", [BATCH, T_pred, IMG_H, IMG_W, IMG_CH])

  # Flatten the images going in s.t. BATCH * T_in, height, width, ch..
  X_flat = tf.reshape(X, [BATCH * T_in, IMG_H, IMG_W, IMG_CH])
  conv1 = conv2d(X_flat, net.weights['wc1'], net.biases['bc1'], 2)
  conv2 = conv2d(conv1, net.weights['wc2'], net.biases['bc2'], 2)
  conv3 = conv2d(conv2, net.weights['wc3'], net.biases['bc3'], 2)

  # Now we hyperflatten everything for the lstm: BATCH, T-in, everything.
  res = tf.reshape(conv3, [BATCH, T_in, -1])
  prediction = net.EncoderDecoder(res)

  # Infer on BATCH * T_pred, everything.
  fc_out = fc(prediction, net.weights['wfc1'], net.biases['bfc1'])
  # Reshape to on BATCH, T_pred, IMG_H * IMG_W * IMG_CH.
  fc_out = tf.reshape(fc_out, [BATCH, T_pred, IMG_H * IMG_W * IMG_CH])
  sig_out = tf.sigmoid(fc_out)

  # Calculate difference..
  Y_flat = tf.reshape(Y, [BATCH, T_pred, IMG_H * IMG_W * IMG_CH])
  diff = fc_out - Y_flat

  # Compute loss...
  loss_op = tf.reduce_sum(tf.reduce_sum(diff * diff, axis=2), axis=1)
  loss_op = tf.reduce_mean(loss_op)
  train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss_op)

  return fc_out, sig_out, X, Y, loss_op, train_op

