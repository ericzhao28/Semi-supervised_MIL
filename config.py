##############################
##### Hyperparameters.
##############################

# TODO: swap to Tensorflow flags
# to align with rest of MIL lib.

DATA_PATH = "/home/ubuntu/data/"
SAVES_PATH = "/home/ubuntu/saves/"

T_in = 16
T_pred = 16
IMG_H = 125
IMG_W = 125
IMG_CH = 3
BATCH = 10
N_EPOCH = 1000
W_STDV = 0.1
LR = 0.001

RNN_H = 512
CONV1_H = 24
CONV2_H = 64
CONV3_H = 64

CONV1_FILTER_H = 10
CONV2_FILTER_H = 10
CONV3_FILTER_H = 10

CONV1_FILTER_W = 10
CONV2_FILTER_W = 10
CONV3_FILTER_W = 10


MODEL_NAME = str("-".join([str(x) for x in {"T_in": T_in,
                  "T_pred": T_pred,
                  "BATCH": BATCH,
                  "LR": LR,
                  "W_STDV": W_STDV,
                  "RNN_H": RNN_H,
                  "CONV1_H": CONV1_H,
                  "CONV2_H": CONV2_H,
                  "CONV3_H": CONV3_H,
                  "CONV1_FILTER_H": CONV1_FILTER_H,
                  "CONV2_FILTER_H": CONV2_FILTER_H,
                  "CONV3_FILTER_H": CONV3_FILTER_H,
                  "CONV1_FILTER_W": CONV1_FILTER_W,
                  "CONV2_FILTER_W": CONV2_FILTER_W,
                  "CONV3_FILTER_W": CONV3_FILTER_W
                 }.values()]))


