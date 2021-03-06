import numpy as np
import random
import tensorflow as tf
import datetime
import dataset
import time

hidden_size = 512
batch_size = 1024
len_per_section = 64

# Training parameters
max_step = 70000
log_every = 100
test_every = 500
checkpoint_directory = 'ckpt'

test_start = "I am thinking that"

text = open('cleaned_posts.txt').read()
data = dataset.DataSet(text)

char_size = data.char_size


def sample(prediction):
    prediction = prediction.flatten()
    r = random.uniform(0, 1)
    s = 0
    char_id = len(prediction) - 1
    for i in range(len(prediction)):
        s += prediction[i]
        if s >= r:
            char_id = i
            break
    char_one_hot = np.zeros(shape=char_size)
    char_one_hot[char_id] = 1.0
    return char_one_hot


def calc_time(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return "%dh, %02dm, %02ds" % (h, m, s)


graph = tf.Graph()
with graph.as_default():
    X = tf.placeholder(tf.float32, [None, None, char_size])
    y = tf.placeholder(tf.float32, [batch_size, char_size])
    input_keep = tf.placeholder(tf.float32)
    output_keep = tf.placeholder(tf.float32)

    global_step = tf.Variable(0)

    weights = {
        'out': tf.Variable(tf.random_normal([1024, char_size]))
    }
    biases = {
        'out': tf.Variable(tf.random_normal([char_size]))
    }

    # Multiple cells need to either be declared like this or in a loop, each
    # by itself!
    cell1 = tf.nn.rnn_cell.BasicLSTMCell(512)
    cell2 = tf.nn.rnn_cell.BasicLSTMCell(1024)
    # cell2 = tf.nn.rnn_cell.DropoutWrapper(
    #     cell2, input_keep_prob=input_keep, output_keep_prob=output_keep)

    rnn_cell = tf.nn.rnn_cell.MultiRNNCell([cell1, cell2])
    outputs, states = tf.nn.dynamic_rnn(rnn_cell, X, dtype=tf.float32)

    outputs = outputs[:, -1, :]

    logits = tf.matmul(outputs, weights['out']) + biases['out']
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))

    optimizer = tf.train.AdamOptimizer(
        0.001).minimize(loss, global_step=global_step)

    prediction = tf.nn.softmax(logits)


with tf.Session(graph=graph) as sess:
    tf.global_variables_initializer().run()
    saver = tf.train.Saver()
    time_start = time.time()
    for step in range(max_step):
        cur_batch = data.next_batch(batch_size, len_per_section)
        _, training_loss = sess.run([optimizer, loss], feed_dict={
                                    X: cur_batch[0], y: cur_batch[1],
                                    input_keep: 0.7, output_keep: 1.0})

        if step % log_every == 0:
            time_now = time.time()
            duration = calc_time(time_now - time_start)
            print("Training loss at step %d: %.2f. Training has taken %s" %
                  (step, training_loss, duration))

            if step % test_every == 0:
                text_generated = test_start

                for i in range(500):
                    text_data = np.zeros((1, len(text_generated), char_size))

                    for idx, char in enumerate(text_generated):
                        text_data[0, idx, data.char2id[char]] = 1.

                    pred_output = sess.run(prediction, feed_dict={
                                           X: text_data, input_keep: 1.0, output_keep: 1.0})
                    likely_char = sample(pred_output)
                    text_generated += data.id2char[np.argmax(likely_char)]

                print('=' * 80)
                print(text_generated)
                print('=' * 80)

                saver.save(sess, checkpoint_directory +
                           '/model', global_step=step)
