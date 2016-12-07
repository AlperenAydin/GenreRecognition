# Author: Alperen AYDIN
# I am going to try out a hybrid network that uses both convolutional and 

import tensorflow as tf
import cnn_functions as cf
from audio_dataset import audio_dataset


dataset = audio_dataset()

#Parameters of the loop
LOG_STEP = 200
SAVER_STEP = 100

# Hyper-parameters of the network
BATCH_SIZE = 10

# The inputs 
x = tf.placeholder(tf.float32, [None, 524288,1,1])
y_ = tf.placeholder(tf.float32, [None, 10])

# We are going to do a series of convolution+MaxPooling to reduce the size of the sound wave

h1 = cf.cnm2x1Layer(x, [7,1,1,3]) # size=262144x3
h2 = cf.cnm2x1Layer(h1, [7,1,3,3]) # size=131072x3

h3 = cf.cnm2x1Layer(h2, [5,1,3,5]) # size=65536x5
h4 = cf.cnm2x1Layer(h3, [5,1,5,5]) # size=32768x5
