import scikits.audiolab as au
import numpy as np

import os

# This is where our data is (I might change this to a more global variable in the 
data_location = '../data/genres'
genres= os.listdir(data_location)
genres.sort() # I am sorting to make sure that the order stays the same regardless of the system

# We are storing the filenames
# But I still want to return them as np arrays so I will filter the output through this function
def load_batch(list_fn):
    data_point, fs, enc = au.auread(list_fn[0])

    data = np.zeros((len(list_fn),600000,1,1), np.float32)

    for i in range(list_fn.shape[0]):
        data_point, fs, enc = au.auread(list_fn[i])
        data[i,:,0,0] = data_point[0:600000]

    return data

class audio_dataset:
    def __init__(self):
        # We cut the data into two
        train_data = []
        train_label = []

        valid_data = []
        valid_label = []
        # Since we don't have the space, we are only storing the 
        for g in genres:
            dirname = os.path.join(data_location,g)
            filenames = os.listdir(dirname)
            filenames.sort()
            filenames = [os.path.join(dirname,fn) for fn in filenames]
            for l in range(len(filenames)):
                if l < 0.8*len(filenames):
                    train_data.append(filenames[l])
                    train_label.append([np.float32(g == genre) for genre in genres])
                else:
                    valid_data.append(filenames[l])
                    valid_label.append([np.float32(g == genre) for genre in genres])

        
        train_data = np.array(train_data)
        valid_data = np.array(valid_data)

        train_label = np.array(train_label)
        valid_label = np.array(valid_label)
    
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

    # Returns the next batch of the TRAINING SET
    # The set is shuffled after each epoch
    def next_batch_train(self,batch_size):
        start = self.index_in_epoch
        self.index_in_epoch += batch_size
        if self.index_in_epoch > self.no_train:
            self.completed_epochs += 1

            #Reshuffle the data
            self.get_new_permutation()
            start = 0
            self.index_in_epoch = batch_size

        end = self.index_in_epoch
        cur_perm = self.perm[start:end]
        return load_batch(self.train_data[cur_perm]), self.train_label[cur_perm], batch_size

    def reset_batch_valid(self):
        self.index_in_valid = 0

    # This returns the next batch of the VALIDATION TEST
    # No shuffling
    def next_batch_valid(self, batch_size):
        start = self.index_in_valid
        self.index_in_valid +=batch_size
        if self.index_in_valid > self.no_valid:
            self.reset_batch_valid()
            return -1, -1, -1
        end = self.index_in_valid
        return load_batch(self.valid_data[start:end]), self.valid_label[start:end], batch_size





        
