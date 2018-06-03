import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from model import LSTMAutoEncoder, build_model
from utils import load_data
from config import T_pred, BATCH, IMG_H, IMG_W, IMG_CH, N_EPOCH, \
    SAVES_PATH, MODEL_NAME


print("Training the model: ", MODEL_NAME)

##############################
##### Load model and data.
##############################
net = LSTMAutoEncoder()
fc_out, sig_out, X, Y, loss_op, train_op = build_model(net)

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
    tf_saver.save(sess, SAVES_PATH + MODEL_NAME,
                  global_step=global_step)

  # Testing the reconstruction .
  batch_X = next(load_data)[0]
  img_pre, img = sess.run([fc_out, sig_out], feed_dict={X: batch_X})
  img_pre = np.reshape(img_pre, [BATCH, T_pred, IMG_H, IMG_W, IMG_CH])
  img = np.reshape(img, [BATCH, T_pred, IMG_H, IMG_W, IMG_CH])

  # Visually validate the results.
  for t in range(T_pred):
    plt.imshow(img_pre[0, t])
    plt.show()

