import tensorflow as tf

from .layers import Conv2d, FC
from .modules import EncoderDecoder


def build(X_shape, t_pred, t_in, img_w, img_h, img_ch, args):
  # Satisfy naming
  model_name = "-".join(vars(args).values())
  print("Building the model: ", model_name)

  # Construct model..
  X = tf.placeholder("float32", X_shape)
  P = tf.reshape(X, [args.batch * t_in] + X_shape[2:])
  Y = tf.placeholder("float32", [args.batch, t_pred] + X_shape[2:])

  P = Conv2d(P, args.batch, args.conv_ch[0], args.conv_st[0],
             args.conv_h[0], args.conv_w[0], args.conv_init[0],
             args.conv_act[0])
  for i in range(len(args.conv) - 1):
    P = Conv2d(P, args.batch, args.conv_ch[i], args.conv_st[i],
               args.conv_h[i], args.conv_w[i], args.conv_init[i],
               args.conv_act[i])

  # Now we hyperflatten everything for the lstm: BATCH, T-in, everything.
  P = tf.reshape(P, (args.batch, t_in, -1))
  P = EncoderDecoder(P, args.batch, args.rnn_init[0], args.rnn[0],
                     args.rnn_cell[0], t_pred)

  # Infer on BATCH * T_pred, everything.
  for i in range(len(args.fc) - 1):
    P = FC(P, args.fc[i], args.fc_init[i], args.fc[i])
  P = FC(P, img_h * img_w * img_ch, args.fc_init[-1], args.fc[-1])

  # Reshape to on BATCH, T_pred, IMG_H * IMG_W * IMG_CH.
  P = tf.reshape(P, (args.batch, t_pred, -1))
  diff = P - tf.reshape(Y, (args.batch, t_pred, -1))

  # Why do we need the sigmoided? Who knows.
  P_norm = tf.sigmoid(P)

  # Compute loss...
  loss_op = tf.reduce_sum(tf.reduce_sum(diff * diff, axis=2), axis=1)
  loss_op = tf.reduce_mean(loss_op)

  train_op = tf.train.AdamOptimizer(learning_rate=args.lr).minimize(loss_op)

  return model_name, P, P_norm, X, Y, loss_op, train_op



