from PIL import Image, ImageSequence
import numpy as np
import pickle
import glob
from multiprocessing import Pool

paths = glob.glob("/home/ubuntu/gym/mil/data/*/*/*.gif")[14800:]
def gif_to_np(path):
  with open(path[:-4] + ".pkl", "wb") as f:
    im = Image.open(path)
data = np.array([np.array(frame.copy().convert("RGB").getdata(), dtype=np.uint8).reshape(frame.size[1], frame.size[0], 3) for frame in ImageSequence.Iterator(im)])
	  pickle.dump(data,f)
	 with Pool(processes=3) as pool:
	 pool.map(gif_to_np, paths, chunksize = 100)

