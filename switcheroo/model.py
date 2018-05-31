import tensorflow as tf
import numpy as np


def build_model(BATCH, T_in, T_pred, net):
  # Construct model
  X = tf.placeholder("float", [BATCH, T_in, 64, 64, 1])
  Y = tf.placeholder("float", [BATCH, T_pred, 64, 64, 1])

  # Conv2D with stride 2 (3 times)
  # Reshape tensor to 4D (Conv2d only supports 4D tensors not 5D)
  X_shape = tf.reshape(X, [BATCH * T_in, 64, 64, 1])
  Y_shape = tf.reshape(Y, [BATCH, T_pred, -1])

  conv1 = conv2d(X_shape, net.weights['wc1'], net.biases['bc1'], 2)
  conv2 = conv2d(conv1, net.weights['wc2'], net.biases['bc2'], 2)
  conv3 = conv2d(conv2, net.weights['wc3'], net.biases['bc3'], 2)

  # Reshape tensor to [BATCH, T, DIM]
  res = tf.reshape(conv3, [BATCH, T_in, -1])

  # LSTM encoders/decoders
  # Input T_in frames, Output T_pred frames
  prediction = net.EncoderDecoder(res)
  print("prediction", prediction.get_shape().as_list())

  # Fully connected: (BATCH*10, 1024) -->(BATCH*10, 4096)
  fc_out = fc(prediction, net.weights['wfc1'], net.biases['bfc1'])
  print("fc_out shape", fc_out.shape)


  # Reshape fc_out to BATCH x T x DIM
  fc_out = tf.reshape(fc_out, [BATCH, T_pred, -1])
  sig_out = tf.sigmoid(fc_out)
  diff = fc_out - Y_shape
  loss_op = 0.5 * tf.reduce_sum(tf.reduce_sum(diff * diff, axis=2), axis=1)
  loss_op = tf.reduce_mean(loss_op)
  train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss_op)
  return fc_out, sig_out, X, Y, loss_op, train_op


def conv2d(x, W, b, strides=1):
  x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
  x = tf.nn.bias_add(x, b)
  return tf.nn.relu(x)


def fc(x, W, b):
  x = tf.matmul(x, W) + b
  return tf.nn.relu(x)


class LSTMAutoEncoder(object):
  def __init__(self, std_dev, batch, t_in, t_pred):
    self.std_dev = std_dev
    self.batch = batch
    self.t_pred = t_pred
    self.t_in = t_in
    self.weights = {
        'wc1': tf.Variable(tf.random_normal([5, 5, 1, 24], stddev=std_dev)),
        'wc2': tf.Variable(tf.random_normal([5, 5, 24, 64], stddev=std_dev)),
        'wc3': tf.Variable(tf.random_normal([5, 5, 64, 64], stddev=std_dev)),
        'wfc1': tf.Variable(tf.random_normal([1024, 4096], stddev=std_dev))
    }

    self.biases = {
        'bc1': tf.Variable(tf.random_normal([24], stddev=0)),
        'bc2': tf.Variable(tf.random_normal([64], stddev=0)),
        'bc3': tf.Variable(tf.random_normal([64], stddev=0)),
        'bfc1': tf.Variable(tf.random_normal([4096], stddev=0))
    }


  def EncoderDecoder(self, data):
    # Encoder Decoder works over 2*T timesteps
    # First phase: Encoding
    # Put in real input data, discard output, keep state
    # Second phase: Decoding
    # Put in zero data (padding), use output/state

    with tf.variable_scope("LSTM") as scope:
      lstm = tf.contrib.rnn.LSTMCell(num_units=1024)
      state = lstm.zero_state(self.batch, "float")
      datum = tf.split(data, self.t_in, axis=1)

      # run lstm for T time step
      for t in range(self.t_in):
        if t > 0:
            scope.reuse_variables()
        output, state = lstm(tf.reshape(datum[t], [self.batch, -1]), state)

      # what is tmp? datum at frame 0, why need to reshape it?
      tmp = tf.reshape(datum[0], [self.batch, -1])

      # Decoding phase
      zero_ = tf.zeros_like(tmp, "float")

      output_list = []
      for t in range(self.t_pred):
        scope.reuse_variables()
        output, state = lstm(zero_, state)
        output_list.append(output)

      out = tf.concat(output_list, axis=1)
      return tf.reshape(out, [self.batch * self.t_pred, -1])

  def load_validation(self):
    file = np.load('/home/fensi/nas/Moving-MNIST/moving-mnist-valid.npz')
    data = file['input_raw_data']
    return data


def load_data():
  # TODO update paths
  file = np.load('/home/fensi/nas/Moving-MNIST/moving-mnist-train.npz')
  data = file['input_raw_data']
  # TODO change input dimensions
  data = np.reshape(data, [-1, 20, 64, 64, 1])
  # TODO update output/in to match config flags and our case
  # Input is the first 10 seconds
  input_seq = data[:, 0:10]
  # Output is after 10 seconds.
  output_seq = data[:, 10:]

  # TODO update paths
  test_file = np.load('/home/fensi/nas/Moving-MNIST/moving-mnist-test.npz')
  test_data = test_file['input_raw_data']
  # TODO change input dimensions
  test_data = np.reshape(test_data, [-1, 20, 64, 64, 1])
  # TODO update output/in to match config flags and our case
  # Input is the first 10 seconds
  test_input_seq = test_data[:, 0:10]
  # Output is after 10 seconds.
  test_output_seq = test_data[:, 10:]
  return input_seq, output_seq, test_input_seq, test_output_seq

