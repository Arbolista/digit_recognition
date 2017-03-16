import json
from neural_net import NeuralNet
from preprocess import Preprocess
import numpy as np

def preprocess_for_length_learner(data, sequence_length=5, side=280):
  images = np.array(data['sequences'])
  labels = np.array(data['labels'])[:, [0]].astype(int) - 1
  return images, labels

p = Preprocess()
d = p.load_file('data/preprocessed_images.pickle')
images = d['trainset']
labels = d['train_labels'].reshape((len(d['train_labels']), 1))

validation_images = d['validset']
validation_labels = d['valid_labels'].reshape((len(d['valid_labels']), 1))

with open('hypes/single_digit.json', 'r') as f:
  H = json.load(f)
  NeuralNet.execute(H, images, labels, validation_images, validation_labels)
