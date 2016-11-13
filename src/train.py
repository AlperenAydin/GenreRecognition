import tensorflow as tf
import cnn_functions as cf
from audio_dataset import audio_dataset


dataset = audio_dataset()
wave,l,bs = dataset.next_batch_valid(10)
print wave.shape

#Parametres of the loop
LOG_STEP = 200
SAVER_STEP = 100

# Hyper-parameters of the network
BATCH_SIZE = 10

x = tf.placeholder(tf.float32, [None, 524288,1,1])
y_ = tf.placeholder(tf.float32, [None, 10])

# We are going to do a series of convolution+MaxPooling to reduce the size of the sound wave

h1 = cf.cnm2x1Layer(x, [7,1,1,3]) # size=262144x3
h2 = cf.cnm2x1Layer(h1, [7,1,3,3]) # size=131072x3

h3 = cf.cnm2x1Layer(h2, [5,1,3,5]) # size=65536x5
h4 = cf.cnm2x1Layer(h3, [5,1,5,5]) # size=32768x5
h5 = cf.cnm2x1Layer(h4, [3,1,5,5]) # size=16384x5
h6 = cf.cnm2x1Layer(h5, [3,1,5,5]) # size=81925x5
h7 = cf.cnm2x1Layer(h6, [3,1,5,5]) # size=4096x5
h8 = cf.cnm2x1Layer(h7, [3,1,5,1]) # size=2048x1

hf = tf.reshape(h8, [-1, 2048])

fc1 = cf.fc_nn(hf,[2048,100])
fc2 = cf.fc_nn(fc1,[100,10])


y = tf.nn.softmax(fc2)

loss = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# We are using the Adam Optimiser (because it is very good at managing the learning rate and momentum. Plus it is very easy to use and I am very llazy)
train_step = tf.train.AdamOptimizer(1e-3).minimize(loss)

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.Session()
sess.run(tf.initialize_all_variables())

with sess.as_default():
    for s in range(1+int(2e6)):
        waves, labels, bs = dataset.next_batch_train(BATCH_SIZE)
        print 'step {}'.format(s)


        train_step.run(feed_dict={x:waves, y_:labels})
