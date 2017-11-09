import numpy as np
import random
import tensorflow as tf
import datetime
import dataset

num_layers = 2
hidden_size = 512
batch_size = 512
len_per_section = 50

# Training parameters
max_step = 70000
log_every = 100
test_every = 100

test_start = "I am thinking that"

text = open('cleaned_posts.txt').read()
data = dataset.DataSet(text)

char_size = data.char_size

graph = tf.Graph()
with graph.as_default():
    X = tf.placeholder(tf.float32, [None, None, char_size])
    y = tf.placeholder(tf.float32, [batch_size, char_size])

    global_step = tf.Variable(0)

    weights = {
        'out': tf.Variable(tf.random_normal([hidden_size, char_size]))
    }
    biases = {
        'out': tf.Variable(tf.random_normal([char_size]))
    }

    cells = []
    for _ in range(num_layers):
        cells.append(tf.nn.rnn_cell.BasicLSTMCell(hidden_size))

    rnn_cell = tf.nn.rnn_cell.MultiRNNCell(cells)
    # outputs, states = tf.nn.dynamic_rnn(rnn_cell, X, dtype=tf.float32)
    x = tf.unstack(X, len_per_section, 1)
    outputs, states = tf.nn.static_rnn(rnn_cell, x, dtype=tf.float32)

    logits = tf.matmul(outputs[-1], weights['out']) + biases['out']
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))

    optimizer = tf.train.AdamOptimizer(
        0.001).minimize(loss, global_step=global_step)


with tf.Session(graph=graph) as sess:
    tf.global_variables_initializer().run()
    for step in range(max_step):
        cur_batch = data.next_batch(batch_size, len_per_section)
        _, training_loss = sess.run([optimizer, loss], feed_dict={
                                    X: cur_batch[0], y: cur_batch[1]})

        if step % log_every == 0:
            print("Training loss at step %d: %.2f" % (step, training_loss))
