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
            output_seq = self.label[self.iter*self.batch_size:(self.iter+1)*self.batch_size]
            
            # extract indexes for this batch
            X1 = self.x1[self.iter*self.batch_size:(self.iter+1)*self.batch_size]
            X2 = self.x2[self.iter*self.batch_size:(self.iter+1)*self.batch_size]  
            # set up y by one-hot vector
            y = np.zeros((self.batch_size, self.numClass))
            y = ku.to_categorical([output_seq], num_classes=self.numClass)
            
            y = np.reshape(y, (y.shape[1], y.shape[2], y.shape[3]))
            self.iter+=1
            
            # stop criteria
            if self.iter>=len(self.x1)//self.batch_size:
                self.iter=0
                np.random.shuffle(self.inds)
            yield [X1, X2], y



def get_hidden_layer_output(lstm, X_train):
    # get hidden layer output
    get_last_hidden_layer_output = K.function([lstm.model.layers[0].input],
                                  [lstm.model.layers[-2].output])
    layer_output = get_last_hidden_layer_output([X_train])[0]
    
    layer_output = np.hstack((layer_output,np.zeros((layer_output.shape[0], 50)))) 
    return layer_output

def define_model(lstm):
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
	# define inference decoder
    decoder_state_input_h = Input(shape=(word_dim,))
    decoder_states_inputs = [decoder_state_input_h]
    gru_1_output =  decoder_gru_1(embedded, initial_state=decoder_states_inputs)
    gru_2_output, state_h = decoder_gru_2(gru_1_output)
    #/decoder_outputs, state_h = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs) # temp ignore state_c
    decoder_states = [state_h]
    decoder_outputs = decoder_dense(gru_2_output)
    inference_model = Model([word_vec_input] + decoder_states_inputs, [decoder_outputs] + decoder_states)
    
    return training_model, inference_model


def predict_sequence(inference_model, X1, X2, n_steps, cardinality):
    #encode
    state = [X2, np.zeros(100)]
    print(X1.shape)
	# start of sequence input
    #target_seq = np.array([0.0 for _ in range(cardinality)]).reshape(1, 1, cardinality)
	# collect predictions
    output = list()
    for t in range(n_steps):
        # predict next char
        yhat, h = inference_model.predict(X1[t] + state)
        # store prediction
        output.append(yhat[0,0,:])
        # update state
        state = [h]
        # update target sequence by next word
        #target_seq = X1[t+1]
    return np.array(output)


def one_hot_decode(encoded_seq):
	return [np.argmax(vector) for vector in encoded_seq]


if __name__ == "__main__":     
    batch_size = 50
    lstm = lstmEncoder(batch_size)
    X_train, y_train, X_val, y_val, X_test, y_test, embedding_matrix = lstm.create_Emb(100)
    
    del y_train, y_val, y_test
    
    lstm.buildModel(embedding_matrix)
    lstm.model.load_weights("./classifier.h5")
    
    layer_output = get_hidden_layer_output(lstm, X_train)
    y = copy.copy(X_train)
    
    
    training_model, inference_model = define_model(lstm)

    # generator
    train_g = Generator(X_train, layer_output, y, lstm.batch_size, lstm.vocab_size)
    training_model.fit_generator(train_g.__getitem__(), steps_per_epoch= math.ceil(len(X_train) / lstm.batch_size), epochs=1)



    # evaluate LSTM
    total, correct = 100, 0
    test_layer_output = get_hidden_layer_output(lstm, X_test[:total])
    y_t = copy.copy(X_test[:total])
    for i in range(total):
        # extract indexes for this batch
        X1 = X_test[i]
        X2 = test_layer_output[i]
        
        y = ku.to_categorical([y_t[i]], num_classes=lstm.vocab_size)
        y = np.reshape(y, (y.shape[1], y.shape[2]))
        
        target = predict_sequence(inference_model, X1, X2, lstm.max_train_len, lstm.vocab_size)
        if np.array_equal(one_hot_decode(y[0]), one_hot_decode(target)):
            correct += 1
    print('Accuracy: %.2f%%' % (float(correct)/float(total)*100.0))
    