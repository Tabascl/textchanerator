import tensorflow as tf
import dataset

text = open('posts.txt').read()
data = dataset.DataSet(text)
char_size = data.char_size

num_layers = 2
hidden_size = 256
batch_size = 512
len_per_section = 50

sess = tf.InteractiveSession()

X = tf.placeholder(tf.float32, [batch_size, len_per_section, char_size])
y = tf.placeholder(tf.float32, [batch_size, char_size])

weights = {
    'out': tf.Variable(tf.random_normal([hidden_size, char_size]))
}
biases = {
    'out': tf.Variable(tf.random_normal([char_size]))
}

cells = []
for _ in range(num_layers):
    cells.append(tf.nn.rnn_cell.BasicLSTMCell(hidden_size))

rnn_cell = tf.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=True)
x = tf.unstack(X, len_per_section, 1)
outputs, states = tf.nn.dynamic_rnn(rnn_cell, X, dtype=tf.float32)

logits = tf.matmul(outputs[-1], weights['out']) + biases['out']

tf.global_variables_initializer().run()

print(sess.run(outputs, feed_dict={X: data.next_batch(batch_size, len_per_section)[0]}))
sess.close()