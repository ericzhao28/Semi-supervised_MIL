import argparse
import tensorflow as tf


def add_args(parser):
  parser.add_argument('-batch', '--batch', type=int, help='batch size',
                      default=64)
  parser.add_argument('-lr', '--lr', type=float, help='learning rate',
                      default=0.001)

  parser.add_argument('-fc', '--fc', type=int, nargs='*',
                      help='hidden units of the feedforward layers',
                      default=[])
  parser.add_argument('-fc_init', '--fc_init', nargs='*',
                      help='initial distribution of fc weights', default=[])
  parser.add_argument('-fc_act', '--fc_act', nargs='*',
                      help='activation function of fc layers', default=[])

  parser.add_argument('-conv_fh', '--conv_fh', type=int, nargs='*',
                      help='filter heights of conv2d layers', default=[])
  parser.add_argument('-conv_fw', '--conv_fw', type=int, nargs='*',
                      help='filter widths of conv2d layers', default=[])
  parser.add_argument('-conv_st', '--conv_st', type=int, nargs='*',
                      help='filter strides of conv2d layers', default=[])
  parser.add_argument('-conv_ch', '--conv_ch', type=int, nargs='*',
                      help='output channels of conv2d layers', default=[])
  parser.add_argument('-conv_init', '--conv_init', nargs='*',
                      help='initial distribution of conv2d weights',
                      default=[])
  parser.add_argument('-conv_act', '--conv_act', nargs='*',
                      help='activation function of conv2d layers', default=[])

  parser.add_argument('-rnn', '--rnn', type=int, nargs='*',
                      help='hidden units of the rnn layers', default=[])
  parser.add_argument('-rnn_cell', '--rnn_cell', nargs='*',
                      help='rnn cell types', default=[])
  parser.add_argument('-rnn_act', '--rnn_act', nargs='*',
                      help='activation function of rnn layers', default=[])
  parser.add_argument('-rnn_init', '--rnn_init', nargs='*',
                      help='initial distribution of rnn weights', default=[])

  return parser


def validate_args(p):
  args = [p.conv_fh, p.conv_fw, p.conv_st, p.conv_ch, p.conv_init, p.conv_act]
  args = [len(x) for x in args]
  assert(args[1:] == args[:-1])

  args = [p.rnn, p.rnn_init, p.rnn_act]
  args = [len(x) for x in args]
  assert(args[1:] == args[:-1])

  args = [p.fc, p.fc_init, p.fc_act]
  args = [len(x) for x in args]
  assert(args[1:] == args[:-1])


def parse_args(p):
  for arg in [p.fc_init, p.fc_act, p.conv_init, p.conv_act,
              p.rnn_act, p.rnn_init]:
    for i, opt in enumerate(arg):
      if opt == "tanh":
        arg[i] = tf.tanh
      elif opt == "relu":
        arg[i] = tf.nn.relu
      elif opt == "sigmoid":
        arg[i] = tf.sigmoid
      elif opt == "xavier":
        arg[i] = tf.contrib.layers.xavier_initializer
      elif opt == "normal":
        arg[i] = tf.random_normal
      elif opt == "lstm":
        arg[i] = tf.contrib.rnn.LSTMCell
      elif opt == "gru":
        arg[i] = tf.contrib.rnn.GRUCell
      else:
        raise ValueError()


parser = argparse.ArgumentParser()
add_args(parser)
args = parser.parse_args()
validate_args(args)
parse_args(args)

