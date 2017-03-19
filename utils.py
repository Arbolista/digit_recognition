from matplotlib import pyplot as plt
import numpy as np
from preprocess import Preprocess
plt.figure()

p = Preprocess()
def get_data():
  return p.load_file('data/test_sequences.pickle')


def show_with_bbox(data, index):
  image = data['sequences'][index]
  label = data['labels'][index]
  length = label[0].astype(int)
  bboxes = label[1:21].reshape(5, 4)

  for i in range(length):
    bbox = bboxes[i]
    xmin = bbox[0]
    xmax = np.min((image.shape[1]-1, xmin+bbox[2]))
    ymin = bbox[1]
    ymax = np.min((image.shape[0]-1, ymin+bbox[3]))
    image[ymin, xmin:xmax] = 1
    image[ymax, xmin:xmax] = 1
    image[ymin:ymax, xmin] = 1
    image[ymin:ymax, xmax] = 1
  plt.imshow(image, cmap='gray')
  plt.show()

