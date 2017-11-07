import numpy as np
import random
import tensorflow as tf
import datetime
import dataset

num_layers = 2
hidden_size = 512
batch_size = 512
len_per_section = 50


text = open('posts.txt').read()
data = dataset.DataSet(text)

char_size = data.char_size

graph = tf.Graph()
with graph.as_default():
    X = tf.placeholder(tf.float32, [batch_size, len_per_section, char_size])
    y = tf.placeholder(tf.float32, [batch_size, char_size])

    weights = {
        'out': tf.Variable(tf.random_normal([hidden_size, char_size]))
    }
    biases = {
        'out': tf.Variable(tf.random_normal([char_size]))
    }

    cells = []
    for i in range(num_layers):
        cells.append(tf.nn.rnn_cell.BasicLSTMCell(hidden_size))

    rnn_cell = tf.nn.rnn_cell.MultiRNNCell(cells)
    outputs, states = tf.nn.static_rnn(rnn_cell, X, sequence_length=batch_size)

    logits = tf.matmul(outputs[-1], weights['out']) + biases['out']
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, ))
