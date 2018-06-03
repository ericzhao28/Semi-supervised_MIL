from PIL import Image, ImageSequence
import numpy as np
import glob
from config import T_in, T_pred, IMG_H, IMG_W, IMG_CH, BATCH


def gif_to_np(path):
  im = Image.open(path)
  data = np.array(
      [np.array(
          frame.copy().convert("RGB").getdata(), dtype=np.uint8
      ).reshape(
          frame.size[1], frame.size[0], 3
      ) for frame in ImageSequence.Iterator(im)]
  )
  return data


def load_single():
  # Get total video size
  vid_l = T_in + T_pred

  # Iterate over all files
  for path in glob.glob("/home/ubuntu/gym/mil/data/*/*/*.gif")[14800:]:
    # Load data.
    data = gif_to_np(path)

    # Skip data where less than T_in + 3 timesteps:
    if data.shape[0] < T_in + 3:
      continue

    # Return typical data points.
    for i in range(0, data.shape[0] - (data.shape[0] % vid_l), vid_l):
      assert(data[i:i + vid_l].shape == (vid_l, IMG_H, IMG_W, IMG_CH))
      yield data[i:i + T_in], data[i + T_in:i + vid_l]

    # Get left overs if sufficient size:
    leftover_size = data.shape[0] % vid_l
    if leftover_size >= T_in + 3:
      leftover = data[:-leftover_size]
      leftover_X = leftover[:T_in]
      leftover_Y = np.zero(T_pred, IMG_H, IMG_W, IMG_CH)
      leftover_Y[:leftover_size - T_in] = leftover[T_in:]

      yield leftover_X, leftover_Y

  return


def load_data():
  # Load single generator.
  singles = load_single()

  while True:
    # Build batches.
    batch_X = np.zero(BATCH, T_in, IMG_H, IMG_W, IMG_CH)
    batch_Y = np.zero(BATCH, T_pred, IMG_H, IMG_W, IMG_CH)
    for i in range(BATCH):
      try:
        batch_X[i], batch_Y[i] = next(singles)
      except StopIteration:
        # End of batch.
        # yield batch
        return
    yield batch_X, batch_Y

