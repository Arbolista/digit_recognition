from __future__ import print_function
from queue import dequeue_and_enqueue
from net_utils import shuffle
import numpy as np
import tensorflow as tf
import datetime
import json

# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
  # Conv2D wrapper, with bias and relu activation
  x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
  x = tf.nn.bias_add(x, b)
  return tf.nn.relu(x)


def maxpool2d(x, k=2):
  # MaxPool2D wrapper
  return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
  padding='SAME')

# Create model
def conv_net(x, config, weights, biases, max_pool_factors):
  # Reshape input picture
  x = tf.reshape(x, shape=[-1, config['height'], config['width'], 1])

  convs = [x]
  # Convolution Layer
  for i in range(len(weights)):
    convi = conv2d(convs[i], weights[i], biases[i])
    if len(max_pool_factors) - 1 >= i and max_pool_factors[i] != 1:
      convi = maxpool2d(convi, k=max_pool_factors[i])
    convs.append(convi)

  return convs[-1]

def fully_connected(x, config, weights, biases, training):
  # Fully connected layer
  # Reshape conv2 output to fit fully connected layer input
  fc = tf.reshape(x, [-1, weights[0].get_shape().as_list()[0]])

  for i in range(len(weights)):
    fc = tf.add(tf.matmul(fc, weights[i]), biases[i])
    if len(config['fully_connected']['relu']) > i and config['fully_connected']['relu'][i]:
      fc = tf.nn.relu(fc)
    if len(config['fully_connected']['dropout']) > i and config['fully_connected']['dropout'][i]:
      train_dropout = tf.constant(config['fully_connected']['dropout'][i])
      fc = tf.select(
        training,
        tf.nn.dropout(fc, train_dropout),
        fc
      )
  return fc


class NeuralNet:

  def __init__(self, config):
    self.config = config

  def make_weights(self):
    config = self.config
    weights = {
    'conv': [],
    'fc': []
    }
    biases = {
    'conv': [],
    'fc': []
    }

    n_in = 1
    if config['conv']:
      for i in range(len(config['conv']['sizes'])):
        weights['conv'].append(
          tf.Variable(tf.random_normal([config['conv']['sizes'][i], config['conv']['sizes'][i], n_in, config['conv']['out'][i]], stddev=config['weights']['stddev']))
        )
        biases['conv'].append(
          tf.Variable(tf.random_normal([config['conv']['out'][i]], stddev=config['biases']['stddev']))
        )
        n_in = config['conv']['out'][i]
      max_pooled_area = reduce(lambda x, y: x/float(np.square(y)), [config['width'] * config['height']] + config['conv']['max_pool_factors'])
      n_in = int(max_pooled_area) * config['conv']['out'][-1]
    else:
      n_in = config['height'] * config['width']

    for i in range(len(config['fully_connected']['out'])):
      weights['fc'].append(
        tf.Variable(tf.random_normal([n_in, config['fully_connected']['out'][i]], stddev=config['weights']['stddev']))
      )
      biases['fc'].append(
        tf.Variable(tf.random_normal([config['fully_connected']['out'][i]], stddev=config['biases']['stddev']))
      )
      n_in = config['fully_connected']['out'][i]
    weights['fc'].append(
      tf.Variable(tf.random_normal([n_in, config['n_outputs'] * config['n_classes']], stddev=config['weights']['stddev']))
    )
    biases['fc'].append(
      tf.Variable(tf.random_normal([config['n_outputs'] * config['n_classes']], stddev=config['biases']['stddev']))
    )
    return weights, biases

  def define(self):
    config = self.config
    with tf.Graph().as_default() as graph:
      self.graph = graph
      weights, biases = self.make_weights()
      self.training = tf.placeholder(tf.bool)

      # tf Graph Input
      self.X = X = tf.placeholder(tf.float32, [None, config['height'], config['width']])

      if config['conv']:
        X1 = conv_net(X, config, weights['conv'], biases['conv'], config['conv']['max_pool_factors'])
        logits = fully_connected(X1, config, weights['fc'], biases['fc'], self.training)
      else:
        logits = fully_connected(X, config, weights['fc'], biases['fc'], self.training)

      n_classes = config['n_classes']
      self.Y = Y = tf.placeholder(tf.int32, [None, config['n_outputs']])
      Yhot = tf.one_hot(Y, n_classes)
      logits = tf.reshape(logits, [-1, config['n_outputs'], n_classes])
      logit_loss = self.logit_loss(logits, Yhot)

      if config['l2_loss']:
        if config['conv']:
          l2_loss = tf.add_n(map(lambda weight: tf.nn.l2_loss(weight), weights['conv']))
        else:
          l2_loss = tf.constant(0.)
        fc_l2_loss = tf.add_n(map(lambda weight: tf.nn.l2_loss(weight), weights['fc']))
        l2_loss = tf.scalar_mul(config['l2_loss']['beta'], tf.add(l2_loss, fc_l2_loss))
      else:
        l2_loss = tf.constant(0.)

      self.cost = cost = logit_loss #tf.add(logit_loss, l2_loss)
      tf.summary.scalar('cost', cost)

      self.pred = pred = tf.argmax(logits, axis=2)

      correct = tf.cast(tf.equal(tf.cast(pred, tf.int32), Y), tf.float32)
      correct_sequence = tf.cast(tf.equal(tf.reduce_sum(correct, 1), config['n_outputs']), tf.float32)

      self.accuracy = accuracy = tf.reduce_mean(correct_sequence)
      self.element_accuracy = element_accuracy = tf.reduce_mean(correct)
      tf.summary.scalar('set_accuracy', accuracy)
      tf.summary.scalar('element_accuracy', element_accuracy)

      self.global_step = global_step = tf.placeholder(tf.int32)

      if config['learning_rate']['decay']:
        learning_rate = tf.train.exponential_decay(config['learning_rate']['start'], global_step, config['iterations'], config['learning_rate']['coefficient'])
      else:
        learning_rate = tf.constant(config['learning_rate']['start'])
      tf.summary.scalar('learning_rate', learning_rate)

      if config['optimizer'] == 'Adam':
        self.optimizer = optimizer = tf.train.AdamOptimizer(learning_rate).minimize(logit_loss)
      elif config['optimizer'] == 'GradientDescent':
        self.optimizer = optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

      # Initializing the variables
      self.init = tf.global_variables_initializer()

      self.summary = tf.summary.merge_all()
      dirname = 'output/%s_%s' % (config['exp_name'],
        datetime.datetime.now().strftime('%Y_%m_%d_%H.%M'))
      self.train_writer = tf.summary.FileWriter(dirname, graph)

      # save hype for later comparisons
      with open(dirname + '/hypes.json', 'w') as f:
          json.dump(config, f, indent=2)

  def logit_loss(self, logits, labels):
    config = self.config
    losses = []
    for i in range(config['n_outputs']):
      logitsi = tf.reshape(tf.slice(logits, [0, i, 0], [-1, 1, -1]), [-1, config['n_classes']])
      labelsi = tf.reshape(tf.slice(labels, [0, i, 0], [-1, 1, -1]), [-1, config['n_classes']])
      lossi = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logitsi, labels=labelsi))
      losses.append(lossi)
    return tf.add_n(losses)

  def train(self, images, labels, validation_images, validation_labels):
    config = self.config
    # Launch the graph
    with tf.Session(graph=self.graph) as sess:
      sess.run(self.init)

      step = 1
      losses = []
      batch_size = config['batch_size']
      # Keep training until reach max iterations
      while step * batch_size < config['iterations']:
        #while (len(losses) <= 3 or losses[-3] > losses[-1]) and step * batch_size < training_iters:
        if (step * batch_size) % float(len(images)) <= batch_size:
          print('shuffling...')
          images, labels = shuffle(images, labels)
        images = dequeue_and_enqueue(images, batch_size)
        batch_x = images[:batch_size]
        labels = dequeue_and_enqueue(labels, batch_size)
        batch_y = labels[:batch_size]
        feed_dict = self.feed_dict(batch_x, batch_y, True, step)
        summary, _ = sess.run([self.summary, self.optimizer], feed_dict=feed_dict)
        self.train_writer.add_summary(summary, step)

        # Display logs per epoch step
        if step % config['summary_step'] == 0:
          feed_dict = self.feed_dict(batch_x, batch_y, False, step)
          summary, c, accuracy, element_accuracy = sess.run([
            self.summary,
            self.cost,
            self.accuracy,
            self.element_accuracy
          ], feed_dict=feed_dict)
          losses.append(c)
          print("Step:", '%04d' % (step), "cost=", "{:.9f}".format(c))
          print("Set Accuracy= {:.9f}".format(accuracy))
          print("Element Accuracy= {:.9f}".format(element_accuracy))
        step += 1
      print("Optimization Finished!")
      feed_dict = self.feed_dict(validation_images, validation_labels, False, 1)
      training_cost, accuracy, pred = sess.run([self.cost, self.accuracy, self.pred], feed_dict=feed_dict)
      print("Validation cost=", training_cost)
      print("Validation accuracy=", accuracy)

  def feed_dict(self, batch_x, batch_y, training, step):
    return {
      self.X: batch_x,
      self.Y: batch_y,
      self.training: training,
      self.global_step: step * self.config['batch_size']
    }

  @classmethod
  def execute(self, config, images, labels, validation_images, validation_labels):
    net = NeuralNet(config)
    net.define()
    net.train(images, labels, validation_images, validation_labels)
