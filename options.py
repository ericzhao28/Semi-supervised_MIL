import tensorflow as tf


BATCH = [100, 200, 50]

LR = [0.001, 0.01, 0.0001]
N_FC = [2, 1, 4]
N_CONVS = [3, 1, 5]
CONVS_FIRST_CHANNELS = [24, 12, 48]
CONVS_CHANNELS = [64, 32, 128]
CONV_STRIDES = [5, 3, 10]
CONV_H = [10, 5, 20]
CONV_W = [10, 5, 20]
CONV_INIT = [tf.contrib.layers.xavier_initializer, tf.random_normal]
CONV_ACT = [tf.relu, tf.tanh, tf.sigmoid]
RNN_ACT = [tf.tanh, tf.sigmoid]
RNN_INIT = [tf.contrib.layers.xavier_initializer, tf.random_normal]
RNN_H = [512, 256, 1024]
RNN_CELL = [tf.contrib.rnn.LSTMCell, tf.contrib.rnn.GRUCell]
FC_HIDDEN = [512, 256, 1024]
FC_INIT = [tf.contrib.layers.xavier_initializer, tf.random_normal]
FC_ACT = [tf.relu, tf.tanh, tf.sigmoid]

