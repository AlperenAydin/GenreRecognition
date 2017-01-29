# Author: Alperen AYDIN
# I am going to try out a hybrid network that uses both convolutional and

import tensorflow as tf
import cnn_functions as cf
from audio_dataset import audio_dataset


dataset = audio_dataset()

# Parameters of the loop
LOG_STEP = 10
SAVER_STEP = 100

# Hyper-parameters of the network
BATCH_SIZE = 1

# The inputs
x = tf.placeholder(tf.float32, [1, 600000, 1, 1])
y_ = tf.placeholder(tf.float32, [None, 10])

X = tf.reshape(x, [10, 60000, 1, 1])
# We are going to do a series of convolution+MaxPooling to reduce the size
# of the sound wave

h1 = cf.cnm2x1Layer(X, [7, 1, 1, 3])   # size=30000x3
h2 = cf.cnm2x1Layer(h1, [7, 1, 3, 3])  # size=15000x3

h3 = cf.cnm2x1Layer(h2, [5, 1, 3, 5])  # size=7500x5
h4 = cf.cnm2x1Layer(h3, [5, 1, 5, 5])  # size=3750x5
h5 = cf.cnm2x1Layer(h4, [3, 1, 5, 1])  # size=1875x1

t = tf.reshape(h5, [-1, 1875, 1])

# Defining the LSTM cell
num_hidden = 15
cell = tf.nn.rnn_cell.LSTMCell(num_hidden, state_is_tuple=True)

val, state = tf.nn.dynamic_rnn(cell, t, dtype=tf.float32)

val = tf.reshape(val, [1, 18750, 15])

# Where are interested in the result we get in the end
# This might be improved
val = tf.transpose(val, [1, 0, 2])
last = tf.gather(val, int(val.get_shape()[0]) - 1)

# We want to map the 24 member output to 10-long vector
fc1 = cf.fc_nn(last, [num_hidden, 10])

# We pass the output through softmax so it represents probabilities
y = tf.nn.softmax(fc1)

# Our loss/energy function is the cross-entropy between the label and the output
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


log = open('logs/rcnn_log.txt', 'a')

saver = tf.train.Saver()
checkpoint = 0

with sess.as_default():
    for s in range(1, int(2e6)):
        waves, labels, bs = dataset.next_batch_train(BATCH_SIZE)
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
                va_x, va_y_, bs = dataset.next_batch_valid(BATCH_SIZE)
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
            log.write(logline)
            print logline

        if s % SAVER_STEP == 0:
            path = saver.save(sess, 'checkpoints/rcnn_',
                              global_step=checkpoint)
            print "Saved checkpoint to %s" % path
            checkpoint += 1

        train_step.run(feed_dict={x: waves, y_: labels})
