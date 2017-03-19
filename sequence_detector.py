import json
from neural_net import NeuralNet
from preprocess import Preprocess
import numpy as np

side = 56
def preprocess_for_length_learner(data, sequence_length=5):
  images = np.array(data['sequences'])
  labels = np.array(data['labels'])[:, (1+sequence_length*4):].astype(int)
  return images, labels

def do_sequence_detector():

  with open('hypes/sequence_detector.json', 'r') as f:
    H = json.load(f)

    p = Preprocess()
    images = np.array([]).reshape(0, side, side)
    labels = np.array([]).reshape(0, 5)
    for i in range(H['dataset_size']):
      data = p.load_file('data/train_sequences{:02d}.pickle'.format(i));
      imagesi , labelsi = preprocess_for_length_learner(data)
      images = np.vstack((images, imagesi))
      labels = np.vstack((labels, labelsi))
    data = None
    labels = labels.astype(int)

    validation_images, validation_labels = preprocess_for_length_learner(p.load_file('data/valid_sequences.pickle'));
    validation_labels = validation_labels.astype(int)


    NeuralNet.execute(H, images, labels, validation_images, validation_labels)
