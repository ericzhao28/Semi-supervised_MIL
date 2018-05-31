import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
from model import LSTMAutoEncoder, build_model, load_data


##############################
##### Hyperparameters.
##############################
# TODO: swap to Tensorflow flags
# to align with rest of MIL lib.
T_in = 10
T_pred = 1
BATCH = 64
num_steps = 1000
std_dev = 0.1
# Number of frames used in
# future prediction, something
# we should optimize.
K = 10

##############################
##### Load model and data.
##############################
net = LSTMAutoEncoder()
inp, out, tst_in, tst_out = load_data()
fc_out, sig_out, X, Y, loss_op, train_op = build_model(BATCH, T_in, T_pred, net)
# Shuffle the sequence
perm = range(inp.shape[0])

##############################
##### Training code.
##############################
with tf.Session() as sess:
  init = tf.global_variables_initializer()
  sess.run(init)

  # Break into epochs.
  for epoch in range(50):
    random.shuffle(perm)
    print("Starting epoch %d" % epoch)
    # Batch data.
    for batch in range(0, inp.shape[0] - (inp.shape[0] % BATCH), BATCH):
      # TODO: Swap 64 to proper image dimensions.
      batch_x = np.zeros([BATCH, T_in, 64, 64, 1])
      batch_y = np.zeros([BATCH, T_pred, 64, 64, 1])
      # Actually load in the data.
      for b in range(BATCH):
        batch_x[b] = inp[perm[start + b], : T_in]
        batch_y[b] = out[perm[start + b], : T_pred]
      # Run TF graph.
      op, loss = sess.run([train_op, loss_op],
                          feed_dict={X: batch_x, Y: batch_y})
      print("Training loss: ", loss)

  # Testing the reconstruction .
  batch_x      = inp[0 : BATCH]
  img_pre, img = sess.run([fc_out, sig_out], feed_dict = {X: batch_x})
  img_pre      = np.reshape(img_pre, [BATCH, T_pred, 64, 64])
  img          = np.reshape(img, [BATCH, T_pred, 64, 64])

  # Visually validate the results.
  for t in range(T_pred):
    plt.imshow(img_pre[0, t])
    plt.show()

