import tensorflow as tf
import cnn_functions as cf

from audio_dataset import audio_dataset

# This code is written after having used this tutorial:
# http://monik.in/a-noobs-guide-to-implementing-rnn-lstm-using-tensorflow/
# There are similarities in the code.

dataset = audio_dataset()


#Paramaters of the training loop
LOG_STEP = 200
SQVER_STEP = 100


# Hyper-parameters of the network
BATCH_SIZE = 10

#Defining the the network

x = tf.placeholder(tf.float32, [None, 524288,1])
y_ = tf.placeholder(tf.float32, [None, 10])

# Defining the LSTM cell
num_hidden = 24
cell = tf.nn.rnn_cell.LSTMCell(num_hidden,state_is_tuple=True)

val, state = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)

# Where are interested in the result we get in the end
# This might be improved
val = tf.transpose(val, [1, 0, 2])
last = tf.gather(val, int(val.get_shape()[0]) - 1)

# We want to map the 24 member output to 10-long vector
fc1 = cf.fc_nn(last, [num_hidden,10])

# We pass the output through softmax so it represents probabilities
y = tf.nn.softmax(fc1)

# Our loss/energy function is the cross-entropy between the label and the output
# We chose this as it offers better results for classification
loss = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# We are using the Adam Optimiser because it is effective at managing the learning rate and momentum
train_step = tf.train.AdamOptimizer(1e-3).minimize(loss)

# Classification accuracy is a better indicator of performance
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.Session()
sess.run(tf.initialize_all_variables())


log = open('rnn/logs/log.txt','a')

saver      = tf.train.Saver()
checkpoint = 0

with sess.as_default():
    for s in range(1+int(2e6)):
        waves, labels, bs = dataset.next_batch_train(BATCH_SIZE)
        waves = waves[:,:,:,0]
        print 'step {}'.format(s)

        # We update the log with the newest performance results
        if (s%LOG_STEP==0):
            # We calculate the performance results
            # for the training set on the current batch
            tr_y = y.eval(feed_dict={x:waves})
            train_loss = loss.eval(feed_dict={y:tr_y, y_:labels})
            train_acc = accuracy.eval(feed_dict={y:tr_y, y_:labels})

            # For the validation set, we do it on the whole thing
            # The final results are means of the results for each batch
            valid_loss = 0
            valid_acc = 0
            batch_count = 0
            while True:
                va_x, va_y_, bs = dataset.next_batch_valid(BATCH_SIZE)
                va_x = va_x[:,:,:,0]
                if bs == -1:
                    break
                batch_count +=1
                
                va_y = y.eval(feed_dict={x:va_x})
                valid_loss += loss.eval(feed_dict={y:va_y, y_:va_y_})
                valid_acc += accuracy.eval(feed_dict={y:va_y, y_:va_y_})
                
            valid_loss = valid_loss/batch_count
            valid_acc = valid_acc/batch_count

            logline = 'Epoch {} Batch {} train_loss {} train_acc {} valid_loss {} valid_acc {} \n'
            logline = logline.format(dataset.completed_epochs, s, train_loss, train_acc, valid_loss, valid_acc)
            log.write(logline)
            print logline

        if s%SAVER_STEP==0:
            path = saver.save(sess, 'rnn/checkpoints/rnn_',global_step=checkpoint)
            print "Saved checkpoint to %s" % path
            checkpoint += 1
            
        train_step.run(feed_dict={x:waves, y_:labels})
