#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# this may be useful, idk lol

from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense, Dropout, TimeDistributed, Activation
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential
import keras.utils as ku
import pickle as pkl
from tensorflow import set_random_seed
import numpy as np
from numpy.random import seed
import string, os

class Generator(object):

    def __init__(self,docs,label,batch_size, numClass):
        self.docs = docs
        self.label = label
        self.batch_size = batch_size
        self.numClass = numClass
        self.inds = np.arange(len(docs))
        self.iter = 0
    def __len__(self):
        return int(np.floor(len(self.docs)/self.batch_size))
    def __getitem__(self):
        while True:
            # init input sequence and output sequence
            input_seq = []
            output_seq = self.label[self.iter*self.batch_size:(self.iter+1)*self.batch_size]
            
            # extract indexes for this batch
            id_list = self.inds[self.iter*self.batch_size:(self.iter+1)*self.batch_size]
            for i in range(self.batch_size):
                input_seq.append(self.docs[id_list[i]])
            
            # set up x by(padding), 
            max_len = max(len(x) for x in input_seq)
            X = pad_sequences(input_seq, maxlen=max_len, padding='pre') 
            #X = pad_sequences(input_seq, maxlen=100, padding='pre') 
            
            # set up y by one-hot vector
            y = np.zeros((self.batch_size, self.numClass))
            for i in range(self.batch_size):
                y[i,:] = ku.to_categorical(output_seq[i], num_classes=self.numClass)
            
            self.iter+=1
            
            # stop criteria
            if self.iter>=len(self.docs)//self.batch_size:
                self.iter=0
                np.random.shuffle(self.inds)
            yield X,y


