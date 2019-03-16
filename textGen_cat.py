#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from keras import backend as K
from keras.layers import Dense, TimeDistributed
from keras.models import Model
from keras.layers import Embedding
from keras.layers import GRU, Input
from lstmEnc_DNN import lstmEncoder 
import keras.utils as ku
import copy
import math

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
            input_seq1 = []
            output_seq = self.label[self.iter*self.batch_size:(self.iter+1)*self.batch_size]
            
            # extract indexes for this batch
            id_list = self.inds[self.iter*self.batch_size:(self.iter+1)*self.batch_size]
            for i in range(self.batch_size):
                input_seq1.append(self.x1[id_list[i]])
            
            # set up x by one-hot vector
            X1 = np.zeros((self.batch_size, self.numClass))
            X1 = ku.to_categorical([input_seq1], num_classes=self.numClass)
            X2 = self.x2[self.iter*self.batch_size:(self.iter+1)*self.batch_size]
               
            # set up y by one-hot vector
            y = np.zeros((self.batch_size, self.numClass))
            y = ku.to_categorical([output_seq], num_classes=self.numClass)

            X1 = np.reshape(X1, (X1.shape[1], X1.shape[2], X1.shape[3]))
            y = np.reshape(y, (y.shape[1], y.shape[2], y.shape[3]))
            
            print(X1.shape, X2.shape, y.shape)
            
            self.iter+=1
            
            # stop criteria
            if self.iter>=len(self.X1)//self.batch_size:
                self.iter=0
                np.random.shuffle(self.inds)
            yield [X1, X2], y




if __name__ == "__main__":     
    batch_size = 50
    lstm = lstmEncoder(batch_size)
    X_train, y_train, X_val, y_val, X_test, y_test, embedding_matrix = lstm.create_Emb()
    
    reshapeX_train = np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1))
    lstm.buildModel(embedding_matrix)
    lstm.model.load_weights("./classifier.h5")
    
    # get hidden layer output
    print("get last hidden layer output")
    get_last_hidden_layer_output = K.function([lstm.model.layers[0].input],
                                  [lstm.model.layers[-2].output])
    layer_output = get_last_hidden_layer_output([X_train])[0]
    
    print(layer_output.shape)
    layer_output = np.hstack((layer_output,np.zeros((layer_output.shape[0], 50))))
    
    # y
    y = copy.copy(X_train)
    
    
    # build seq2seq model
    word_dim = 100
    
    word_vec_input = Input(shape=(lstm.max_train_len,))
    hiddenLayer_inputs = Input(shape=(100,))
    decoder_embed = Embedding(input_dim=lstm.vocab_size, output_dim=word_dim, weights=[embedding_matrix], input_length=None, trainable=False)
    decoder_gru_1 = GRU(word_dim, return_sequences=True, return_state=False)
    decoder_gru_2 = GRU(word_dim, return_sequences=True, return_state=True)
    decoder_dense = Dense(lstm.vocab_size, activation='softmax')
    
    embedded = decoder_embed(word_vec_input)
    gru_1_output = decoder_gru_1(embedded, initial_state=hiddenLayer_inputs)
    gru_2_output, state_h = decoder_gru_2(gru_1_output)
    decoder_outputs = decoder_dense(gru_2_output)
    
    # Define the model that will be used for training
    training_model = Model([word_vec_input, hiddenLayer_inputs], decoder_outputs)
    training_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    print(training_model.summary())
    
    # Also create a model for inference (this returns the GRU state)
    decoder_model = Model([word_vec_input, hiddenLayer_inputs], [decoder_outputs, state_h])
    
    # generator
    train_g = Generator(X_train, layer_output, y, lstm.batch_size, lstm.vocab_size)
    training_model.fit_generator(train_g.__getitem__(), steps_per_epoch= math.ceil(len(X_train) / lstm.batch_size), epochs=1)
    #training_model.fit([X_train, layer_output], reshapeX_train, epochs=10, batch_size = lstm.batch_size);
    