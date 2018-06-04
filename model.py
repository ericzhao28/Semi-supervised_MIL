from layers import Conv2d, EncoderDecoder, FC
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from utils import load_data
from config import T_in, T_pred, IMG_H, IMG_W, IMG_CH, N_EPOCH, SAVES_PATH


def build(X_shape, t_pred, hyperparameters):
  LR, N_FC, N_CONVS, CONV_FIRST_CHANNELS, CONV_CHANNELS, CONV_STRIDES, \
      CONV_H, CONV_W, CONV_INIT, CONV_ACT, RNN_INIT, RNN_H, RNN_CELL, \
      FC_HIDDEN, FC_INIT, FC_ACT = hyperparameters

  model_name = "-".join(hyperparameters)

  batch_s, t_in, img_h, img_w, img_ch = X_shape

  # Satisfy naming
  print("Building the model: ", model_name)

  # Construct model..
  X = tf.placeholder("float32", X_shape)
  P = tf.reshape(X, [batch_s * t_in] + X_shape[2:])
  Y = tf.placeholder("float32", [batch_s, t_pred] + X_shape[2:])

  P = Conv2d(P, batch_s, CONV_FIRST_CHANNELS, CONV_STRIDES,
             CONV_H, CONV_W, CONV_INIT, CONV_ACT)
  for i in range(N_CONVS - 1):
    P = Conv2d(P, batch_s, CONV_CHANNELS, CONV_STRIDES,
               CONV_H, CONV_W, CONV_INIT, CONV_ACT)

  # Now we hyperflatten everything for the lstm: BATCH, T-in, everything.
  P = tf.reshape(P, (batch_s, t_in, -1))
  P = EncoderDecoder(P, batch_s, RNN_INIT, RNN_H, RNN_CELL, t_pred)

  # Infer on BATCH * T_pred, everything.
  for i in range(N_FC - 1):
    P = FC(P, FC_HIDDEN, FC_INIT, FC_ACT)
  P = FC(P, img_h * img_w * img_ch, FC_INIT, FC_ACT)

  # Reshape to on BATCH, T_pred, IMG_H * IMG_W * IMG_CH.
  P = tf.reshape(P, (batch_s, t_pred, -1))
  diff = P - tf.reshape(Y, (batch_s, t_pred, -1))

  # Why do we need the sigmoided? Who knows.
  P_norm = tf.sigmoid(P)

  # Compute loss...
  loss_op = tf.reduce_sum(tf.reduce_sum(diff * diff, axis=2), axis=1)
  loss_op = tf.reduce_mean(loss_op)

  train_op = tf.train.AdamOptimizer(learning_rate=LR).minimize(loss_op)

  return model_name, P, P_norm, X, Y, loss_op, train_op


def train(hyperparameters):
  batch_s = hyperparameters[0]
  hyperparameters = hyperparameters[1:]

  ##############################
  ##### Load model and data.
  ##############################
  X_shape = (batch_s, T_in, IMG_H, IMG_W, IMG_CH)
  model_name, fc_out, sig_out, X, Y, loss_op, train_op = build(
      X_shape, T_pred, hyperparameters)
  global_step = tf.Variable(0,
                            dtype=tf.int32,
                            trainable=False,
                            name='global_step')
  inc_global_step = tf.assign_add(global_step, 1, name="increment")

  ##############################
  ##### Training code.
  ##############################
  with tf.Session() as sess:
    # Variable initialization.
    init = tf.global_variables_initializer()
    sess.run(init)

    # Build saver
    tf_saver = tf.train.Saver(tf.global_variables())

    # Optional restore
    # tf_saver.restore(sess, SAVES_PATH + MODEL_NAME)

    # Break into epochs.
    for epoch in range(N_EPOCH):
      print("Starting epoch %d" % epoch)
      # Batch data.
      for batch_X, batch_Y in load_data():
        # Run TF graph.
        op, loss = sess.run([train_op, loss_op],
                            feed_dict={X: batch_X, Y: batch_Y})
        sess.run(inc_global_step)
        print("Training loss: ", loss)

      # Save weights
      print("Saving...")
      tf.train.global_step(sess, global_step)
      tf_saver.save(sess, SAVES_PATH + model_name,
                    global_step=global_step)

    # Testing the reconstruction .
    batch_X = next(load_data)[0]
    img_pre, img = sess.run([fc_out, sig_out], feed_dict={X: batch_X})
    img_pre = np.reshape(img_pre, [batch_s, T_pred, IMG_H, IMG_W, IMG_CH])
    img = np.reshape(img, [batch_s, T_pred, IMG_H, IMG_W, IMG_CH])

    # Visually validate the results.
    for t in range(T_pred):
      plt.imshow(img_pre[0, t])
      plt.show()

