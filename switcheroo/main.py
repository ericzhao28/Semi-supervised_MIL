import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
from model import LSTMAutoEncoder, build_model, load_data
from config import T_in, T_pred, BATCH, IMG_H, IMG_W, IMG_CH


##############################
##### Load model and data.
##############################
net = LSTMAutoEncoder()
train_X, train_Y = load_data()
fc_out, sig_out, X, Y, loss_op, train_op = build_model(net)

perm = range(train_X.shape[0])
random.shuffle(perm)

##############################
##### Training code.
##############################
with tf.Session() as sess:
  # Variable initialization.
  init = tf.global_variables_initializer()
  sess.run(init)

  # Break into epochs.
  for epoch in range(N_EPOCH):
    print("Starting epoch %d" % epoch)
    # Batch data.
    for batch in range(0, X.shape[0] - (X.shape[0] % BATCH), BATCH):
      # TODO: Swap 64 to proper image dimensions.
      batch_x = np.zeros([BATCH, T_in, IMG_H, IMG_W, IMG_CH])
      batch_y = np.zeros([BATCH, T_pred, IMG_H, IMG_W, IMG_CH])
      # Actually load in the data.
      for b in range(BATCH):
        batch_x[b] = train_X[perm[start + b], : T_in]
        batch_y[b] = train_Y[perm[start + b], : T_pred]
      # Run TF graph.
      op, loss = sess.run([train_op, loss_op],
                          feed_dict={X: batch_x, Y: batch_y})
      print("Training loss: ", loss)

  # Testing the reconstruction .
  batch_x = train_X[0 : BATCH]
  img_pre, img = sess.run([fc_out, sig_out], feed_dict = {X: batch_x})
  img_pre = np.reshape(img_pre, [BATCH, T_pred, IMG_H, IMG_W, IMG_CH])
  img = np.reshape(img, [BATCH, T_pred, IMG_H, IMG_W, IMG_CH])

  # Visually validate the results.
  for t in range(T_pred):
    plt.imshow(img_pre[0, t])
    plt.show()

