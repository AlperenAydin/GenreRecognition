# Author: Alperen AYDIN
# I am going to try out a hybrid network that uses both convolutional and

import tensorflow as tf
import cnn_functions as cf
from audio_dataset import audio_dataset


dataset = audio_dataset()

# Parameters of the loop
LOG_STEP = 10
SAVER_STEP = 50


# Hyper parametres of the network

num_segments = 64
length = dataset.max_length / num_segments
# The inputs
x = tf.placeholder(tf.float32, [1, dataset.max_length, 1])
y_ = tf.placeholder(tf.float32, [1, 10])

X = tf.reshape(x, [num_segments, length, 1])

# Defining the first LSTM cell
num_hidden = 16
with tf.variable_scope('first_LSTM'):
    cell_1 = tf.nn.rnn_cell.LSTMCell(num_hidden, state_is_tuple=True)

    out, state_1 = tf.nn.dynamic_rnn(cell_1, X, dtype=tf.float32)

    out = tf.reshape(out, [1, num_segments, length,
                           num_hidden])  # 1x64xlengthx16

# We are going to do a series of convolution+MaxPooling to reduce the size
# of the sound wave

h1 = cf.cnm2x2Layer(out, [11, 3, num_hidden, 8])  # 1x32x(length/2)x8
h2 = cf.cnm2x2Layer(h1,  [5, 3, 8, 4])    # 1x16x(length/4)x8
h3 = cf.cnm2x2Layer(h2,  [3, 3, 4, 2])    # 1x8x(length/8)x4
h4 = cf.cnm2x2Layer(h3,  [3, 3, 2, 1])    # 1x4x(length/16)x1

# Entering the second LSTM cell

H = tf.reshape(h4, [1, length * num_segments / (2**(4 * 2)), 1])

num_hidden = 8

with tf.variable_scope('second_LSTM'):
    cell_2 = tf.nn.rnn_cell.LSTMCell(num_hidden, state_is_tuple=True)
    val, state_2 = tf.nn.dynamic_rnn(cell_2, H, dtype=tf.float32)

# Where are interested in the result we get in the end
# This might be improved
val = tf.transpose(val, [1, 0, 2])
last = tf.gather(val, int(val.get_shape()[0]) - 1)

# We want to map the 24 member output to 10-long vector
fc1 = cf.fc_nn(last, [num_hidden, 10])

# We pass the output through softmax so it represents probabilities
y = tf.nn.softmax(fc1)

# Our loss/energy function is the cross-entropy
# between the label and the output

# We chose this as it offers better results for classification
loss = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# We are using the Adam Optimiser because it is effective at managing the
# learning rate and momentum
train_step = tf.train.AdamOptimizer(1e-3).minimize(loss)

# Classification accuracy is a better indicator of performance
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.Session()
sess.run(tf.initialize_all_variables())


saver = tf.train.Saver()
checkpoint = 0

with sess.as_default():
    for s in range(int(2e5)):
        waves, labels, bs = dataset.next_batch_train()
        print 'step {}'.format(s)

        # We update the log with the newest performance results
        if (s % LOG_STEP == 0):
            # We calculate the performance results
            # for the training set on the current batch
            tr_y = y.eval(feed_dict={x: waves})
            train_loss = loss.eval(feed_dict={y: tr_y, y_: labels})
            train_acc = accuracy.eval(feed_dict={y: tr_y, y_: labels})

            # For the validation set, we do it on the whole thing
            # The final results are means of the results for each batch
            valid_loss = 0
            valid_acc = 0
            batch_count = 0.0
            while True:
                va_x, va_y_, bs = dataset.next_batch_valid()
                if bs == -1:
                    break
                batch_count += 1.0

                va_y = y.eval(feed_dict={x: va_x})
                valid_loss += loss.eval(feed_dict={y: va_y, y_: va_y_})
                valid_acc += accuracy.eval(feed_dict={y: va_y, y_: va_y_})

            valid_loss = valid_loss / batch_count
            valid_acc = (valid_acc) / batch_count
            # Adding a new line to the log
            logline = 'Epoch {} Batch {} train_loss {} train_acc {} valid_loss {} valid_acc {} \n'
            logline = logline.format(
                dataset.completed_epochs, s, train_loss, train_acc, valid_loss, valid_acc)
            log = open('logs/rcnn_log.txt', 'a')
            log.write(logline)
            log.close()
            print logline

        if s % SAVER_STEP == 0:
            path = saver.save(sess, 'checkpoints/rcnn_',
                              global_step=checkpoint)
            print "Saved checkpoint to %s" % path
            checkpoint += 1

        train_step.run(feed_dict={x: waves, y_: labels})
