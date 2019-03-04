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

    def __init__(self,data,batch_size,vocab_size):
        self.data = data
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.inds = np.arange(len(data))
        self.iter = 0
    def __len__(self):
        return int(np.floor(len(self.data)/self.batch_size))
    def __getitem__(self):
        while True:
            input_seq = []
            id_list = self.inds[self.iter*self.batch_size:(self.iter+1)*self.batch_size]
            for i in range(self.batch_size):
                input_seq.append(self.data[id_list[i]])
            max_len = max(len(x) for x in input_seq)
            padded = pad_sequences(input_seq, maxlen=max_len, padding='pre')
            X = np.array(padded)[:,:-1]
            y = np.zeros((self.batch_size,max_len-1,self.vocab_size))
            for i in range(self.batch_size):
                y[i,:,:] = ku.to_categorical(padded[i][1:], num_classes=self.vocab_size)
            self.iter+=1
            if self.iter>=len(self.data)//self.batch_size:
                self.iter=0
                np.random.shuffle(self.inds)
            yield X,y


