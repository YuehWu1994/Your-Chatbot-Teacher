
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# this may be useful, idk lol

from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense, Dropout, TimeDistributed, Activation
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential
import keras.utils as ku
from tensorflow import set_random_seed
import numpy as np
from numpy.random import seed

import pickle as pkl

class Generator(object):

    def __init__(self, x1, x2, label,batch_size, numClass):
        self.x1 = x1
        self.x2 = x2
        self.label = label
        self.batch_size = batch_size
        self.numClass = numClass
        self.inds = np.arange(len(x1))
        self.iter = 0
    def __len__(self):
        return int(np.floor(len(self.x1)/self.batch_size))
    def __getitem__(self):
        while True:
            # init input sequence and output sequence
            input_seq1, input_seq2 = [], []
            output_seq = self.label[self.iter*self.batch_size:(self.iter+1)*self.batch_size]
            
            # extract indexes for this batch
            id_list = self.inds[self.iter*self.batch_size:(self.iter+1)*self.batch_size]
            for i in range(self.batch_size):
                input_seq1.append(self.x1[id_list[i]])
                input_seq2.append(self.x2[id_list[i]])
            
            # set up x by one-hot vector
            x1 = np.zeros((self.batch_size, self.numClass))
            x1 = ku.to_categorical([input_seq1], num_classes=self.numClass)
            x2 = np.zeros((self.batch_size, self.numClass))
            x2 = ku.to_categorical([input_seq2], num_classes=self.numClass)
            
            # set up y by one-hot vector
            y = np.zeros((self.batch_size, self.numClass))
            y = ku.to_categorical([output_seq], num_classes=self.numClass)

            x1 = np.reshape(x1, (x1.shape[1], x1.shape[2], x1.shape[3]))
            x2 = np.reshape(x2, (x2.shape[1], x2.shape[2], x2.shape[3]))
            y = np.reshape(y, (y.shape[1], y.shape[2], y.shape[3]))
            
            self.iter+=1
            
            # stop criteria
            if self.iter>=len(self.x1)//self.batch_size:
                self.iter=0
                np.random.shuffle(self.inds)
            yield [x1, x2], y


