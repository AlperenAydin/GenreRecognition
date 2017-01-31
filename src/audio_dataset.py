import scikits.audiolab as au
import numpy as np

import os
import math

# IMPORTANT NOTE: This code is designed to work on my 4GB laptop
# When we get to have AWS instances with bigger RAM,
# I will design a version which loads everything into memory

# This is where our data is (I might change this to a more global variable
# in the
data_location = '../data/genres'
genres = os.listdir(data_location)
genres.sort()
# I am sorting to make sure that the order stays the same regardless of
# the system


class audio_dataset:

    def __init__(self):
        # We cut the data into two parts
        # Since we do not have enough information we will do 80% train and 20%
        train_data = []
        train_label = []

        valid_data = []
        valid_label = []
        # Since we don't have the space,
        # we are only loading the filenames in memory
        # We will use load_batch() above to load the audio files when we need
        # them.
        self.max_length = 0
        for g in genres:
            dirname = os.path.join(data_location, g)
            filenames = os.listdir(dirname)
            filenames.sort()
            filenames = [os.path.join(dirname, fn) for fn in filenames]
            for l in range(len(filenames)):
                file, _, _ = au.auread(filenames[l])
                if(self.max_length < file.shape[0]):
                    self.max_length = file.shape[0]
                data = filenames[l]
                label = np.array([np.float32(g == genre) for genre in genres])
                if l < 0.8 * len(filenames):
                    train_data.append(data)
                    train_label.append(label)
                else:
                    valid_data.append(data)
                    valid_label.append(label)
        self.max_length = 2**(int(math.log(self.max_length, 2)))
        train_data = np.array(train_data)
        valid_data = np.array(valid_data)

        train_label = np.array(train_label)
        valid_label = np.array(valid_label)

        # Setting a bunch of variables
        self.no_train = train_data.shape[0]
        self.no_valid = valid_data.shape[0]
        self.no_classes = valid_label.shape[1]
        self.train_data = train_data
        self.train_label = train_label
        self.valid_data = valid_data
        self.valid_label = valid_label

        self.index_in_epoch = 0
        self.index_in_valid = 0
        self.completed_epochs = 0

        self.get_new_permutation()

    # We assume that the data will be fed randomly
    # We randomize the output of the train data
    def get_new_permutation(self):
        self.perm = np.arange(self.no_train)
        np.random.shuffle(self.perm)

    # We are storing the filenames
    # But I still want to return them as np arrays so I will filter the output
    # through this function
    def load_batch(self, filename):
        data = np.zeros((1, self.max_length, 1), np.float32)
        data_point, fs, enc = au.auread(filename)
        data[0, 0:self.max_length, 0] = data_point[0:self.max_length]
        return data

    # Returns the next batch of the TRAINING SET
    # The set is shuffled after each epoch
    def next_batch_train(self):
        start = self.index_in_epoch
        self.index_in_epoch += 1
        if self.index_in_epoch > self.no_train:
            self.completed_epochs += 1

            # Reshuffle the data
            self.get_new_permutation()
            start = 0
            self.index_in_epoch = 0

        cur_perm = self.perm[start]
        return (self.load_batch(self.train_data[cur_perm]),
                np.reshape(self.train_label[cur_perm], (1, 10)),
                1)

    def reset_batch_valid(self):
        self.index_in_valid = 0

    # This returns the next batch of the VALIDATION TEST
    # No shuffling
    def next_batch_valid(self):
        start = self.index_in_valid
        self.index_in_valid += 1
        if self.index_in_valid > self.no_valid:
            self.reset_batch_valid()
            return -1, -1, -1
        return (self.load_batch(self.valid_data[start]),
                np.reshape(self.valid_label[start], (1, 10)),
                1)
