from model import train
from options import BATCH, LR, N_FC, N_CONVS, CONV_FIRST_CHANNELS, \
    CONV_CHANNELS, CONV_STRIDES, CONV_H, CONV_W, CONV_INIT, \
    CONV_ACT, RNN_INIT, RNN_H, RNN_CELL, FC_HIDDEN, FC_INIT, FC_ACT


test_params = []
for x in [BATCH, LR, N_FC, N_CONVS, CONV_FIRST_CHANNELS,
          CONV_CHANNELS, CONV_STRIDES, CONV_H, CONV_W,
          CONV_INIT, CONV_ACT, RNN_INIT, RNN_H,
          RNN_CELL, FC_HIDDEN, FC_INIT, FC_ACT]:
  test_params.append(x[0])

train(test_params)
