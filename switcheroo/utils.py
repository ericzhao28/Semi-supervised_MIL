from PIL import Image, ImageSequence
import numpy as np
import pickle
import glob


paths = glob.glob("/home/ubuntu/gym/mil/data/*/*/*.gif")[14800:]
def gif_to_np(path):
  with open(path[:-4] + ".pkl", "wb") as f:
    im = Image.open(path)
    data = np.array(
        [np.array(
            frame.copy().convert("RGB").getdata(), dtype=np.uint8
         ).reshape(
            frame.size[1], frame.size[0], 3
         ) for frame in ImageSequence.Iterator(im)]
    )
	  pickle.dump(data,f)
    with Pool(processes=3) as pool:
	    pool.map(gif_to_np, paths, chunksize = 100)


def load_data():
  # TODO change input dimensions
  data = np.reshape(data, [-1, VID_L, IMG_H, IMG_W, IMG_CH])
  # TODO update output/in to match config flags and our case
  # Input is the first 10 seconds
  X = data[:, 0:10]
  # Output is after 10 seconds.
  Y = data[:, 10:]

  return X, Y

