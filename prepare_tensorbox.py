from six.moves import cPickle as pickle
import json
import cv2

def load_pickle(filename):
  with open(filename, 'rb') as f:
    return pickle.load(f)

class PrepareTensorbox:

  @classmethod
  def jpg(self):
    print('valid sequences....')
    d = load_pickle('data/valid_sequences.pickle')
    for i in range(len(d['sequences'])):
      filename = 'tensorbox/sequence_images/valid_sequence-{:04d}.jpg'.format(i)
      image = d['sequences'][i]
      cv2.imwrite(filename, image*255)
    d = load_pickle('data/test_sequences.pickle')
    print('test sequences...')
    for i in range(len(d['sequences'])):
      filename = 'tensorbox/sequence_images/test_sequence-{:04d}.jpg'.format(i)
      image = d['sequences'][i]
      cv2.imwrite(filename, image*255)
    print('train sequences....')
    for i in range(11):
      d = load_pickle('data/train_sequences{:02d}.pickle'.format(i))
      for j in range(len(d['sequences'])):
        filename = 'tensorbox/sequence_images/sequence-{:02d}-{:04d}.jpg'.format(i, j)
        image = d['sequences'][j]
        cv2.imwrite(filename, image*255)


  @classmethod
  def image_rects(self, label):
    rects = []
    sequence_length = label[0]
    bboxes = label[1:(sequence_length * 4 + 1)].reshape(sequence_length, 4)
    for bbox in bboxes:
      rect = {
        'x1': bbox[0],
        'x2': bbox[0] + bbox[2],
        'y1': bbox[1],
        'y2': bbox[1] + bbox[3]
      }
      rects.append(rect)
    return rects

  @classmethod
  def json(self):

    print('valid sequences....')
    d = load_pickle('data/valid_sequences.pickle')
    data = []
    for i in range(len(d['sequences'])):
      image_data = {}
      image_data['image_path'] = 'sequence_images/valid_sequence-{:04d}.jpg'.format(i)
      image_data['rects'] = PrepareTensorbox.image_rects(d['labels'][i])
      data.append(image_data)
    with open('tensorbox/data/valid_idl.json', 'w') as outfile:
        json.dump(data, outfile)

    print('test sequences...')
    data = []
    d = load_pickle('data/test_sequences.pickle')
    for i in range(len(d['sequences'])):
      image_data = {}
      image_data['image_path'] = 'sequence_images/valid_sequence-{:04d}.jpg'.format(i)
      image_data['rects'] = PrepareTensorbox.image_rects(d['labels'][i])
      data.append(image_data)
    with open('tensorbox/data/test_idl.json', 'w') as outfile:
        json.dump(data, outfile)

    print('train sequences....')
    data = []
    for i in range(11):
      d = load_pickle('data/train_sequences{:02d}.pickle'.format(i))
      for j in range(len(d['sequences'])):
        image_data = {}
        image_data['image_path'] = 'sequence_images/sequence-{:02d}-{:04d}.jpg'.format(i, j)
        image_data['rects'] = PrepareTensorbox.image_rects(d['labels'][j])
        data.append(image_data)
    with open('tensorbox/data/train_idl.json', 'w') as outfile:
        json.dump(data, outfile)

PrepareTensorbox.json()
