import json
from neural_net import NeuralNet
from preprocess import Preprocess
import numpy as np

def preprocess_for_length_learner(data, sequence_length=5, side=280):
  images = np.array(data['sequences'])
  labels = np.array(data['labels'])[:, [0]].astype(int) - 1
  return images, labels

p = Preprocess()
images = np.array([]).reshape(0, 280, 280)
labels = np.array([]).reshape(0, 1)
for i in range(11):
  data = p.load_file('data/train_sequences{:02d}.pickle'.format(i));
  imagesi , labelsi = preprocess_for_length_learner(data)
  images = np.vstack((images, imagesi))
  labels = np.vstack((labels, labelsi))
data = None
labels = labels.astype(int)

validation_images, validation_labels = preprocess_for_length_learner(p.load_file('data/valid_sequences.pickle'));
validation_labels = validation_labels.astype(int)

with open('hypes/length.json', 'r') as f:
  H = json.load(f)
  NeuralNet.execute(H, images, labels, validation_images, validation_labels)
