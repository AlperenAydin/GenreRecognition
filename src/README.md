# Source code:
==============

Here we have the source code for our network.


# Convolutional Neural Network:
===============================

Right now, our network is a simple convolutional neural network. It takes in the 2**19 long vector and reduces it to 2**11 which we then feed to 2 fully conncected neural network layer.

# Recurrent Network:
====================

The above example isn't very good for this type of job. The best would be a recurrent network. This type of network is implemented in the rnn_train.py script.

It takes in the same vector as the CNN but this network has only a single lstm cell followed by fully connected layer that maps the output of the lstm to out 1*10 output. 