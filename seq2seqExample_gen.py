#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from random import randint
import numpy as np
from keras import backend as K
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import GRU, Input

from seq2seq_generator import Generator as gen
from lstmEnc_DNN import lstmEncoder 
import math
import copy 


# generate a sequence of random integers
def generate_sequence(length, n_unique):
	return [randint(1, n_unique-1) for _ in range(length)]

# prepare data for the LSTM
def get_dataset(n_in, n_out, cardinality, n_samples):
	X1, X2, y = list(), list(), list()
	for _ in range(n_samples):
		# generate source sequence
		source = generate_sequence(n_in, cardinality)
		# define padded target sequence
		target = source[:n_out]
		target.reverse()
		# create padded input target sequence
		target_in = [0] + target[:-1]
		# encode
		src_encoded = source
		tar_encoded = target
		tar2_encoded = target_in
		# store
		X1.append(src_encoded)
		X2.append(tar2_encoded)
		y.append(tar_encoded)
    
	X1 = np.array(X1)
	X2 = np.array(X2)
	y = np.array(y)     

	return np.array(X1), np.array(X2), np.array(y)

def categorical(X1, X2, y, cardinality):
	X1 = to_categorical([X1], num_classes=cardinality)
	X2 = to_categorical([X2], num_classes=cardinality)
	y = to_categorical([y], num_classes=cardinality)

	X1 = np.reshape(X1, (X1.shape[1], X1.shape[2], X1.shape[3]))
	X2 = np.reshape(X2, (X2.shape[1], X2.shape[2], X2.shape[3]))
	y = np.reshape(y, (y.shape[1], y.shape[2], y.shape[3]))
	return X1, X2, y		
		
    

# returns train, inference_encoder and inference_decoder models
def define_models(n_input, n_output, n_units):
	# define training encoder
	encoder_inputs = Input(shape=(None, n_input))
	encoder = LSTM(n_units, return_state=True)
	encoder_outputs, state_h, state_c = encoder(encoder_inputs)
	encoder_states = [state_h, state_c]
	# define training decoder
	decoder_inputs = Input(shape=(None, n_output))
	decoder_lstm = LSTM(n_units, return_sequences=True, return_state=True)
	decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
	decoder_dense = Dense(n_output, activation='softmax')
	decoder_outputs = decoder_dense(decoder_outputs)
	model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
	# define inference encoder
	encoder_model = Model(encoder_inputs, encoder_states)
	# define inference decoder
	decoder_state_input_h = Input(shape=(n_units,))
	decoder_state_input_c = Input(shape=(n_units,))
	decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
	decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
	decoder_states = [state_h, state_c]
	decoder_outputs = decoder_dense(decoder_outputs)
	decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
	# return all models
	return model, encoder_model, decoder_model

# generate target given source sequence
def predict_sequence(infenc, infdec, source, n_steps, cardinality):
	# encode
	state = infenc.predict(source)
	# start of sequence input
	target_seq = np.array([0.0 for _ in range(cardinality)]).reshape(1, 1, cardinality)
	# collect predictions
	output = list()
	for t in range(n_steps):
		# predict next char
		yhat, h, c = infdec.predict([target_seq] + state)
		# store prediction
		output.append(yhat[0,0,:])
		# update state
		state = [h, c]
		# update target sequence
		target_seq = yhat
	return np.array(output)

# decode a one hot encoded string
def one_hot_decode(encoded_seq):
	return [np.argmax(vector) for vector in encoded_seq]


if __name__ == "__main__":   

    batch_size = 50
    lstm = lstmEncoder(batch_size)
    X_train, y_train, X_val, y_val, X_test, y_test, embedding_matrix = lstm.create_Emb()
    
    lstm.buildModel(embedding_matrix)
    lstm.model.load_weights("./classifier.h5")
    
    # get hidden layer output
    print("get last hidden layer output")
    get_last_hidden_layer_output = K.function([lstm.model.layers[0].input],
                                  [lstm.model.layers[-2].output])
    layer_output = get_last_hidden_layer_output([X_train])[0]
    
    print(layer_output.shape)
    layer_output = np.hstack((layer_output,np.zeros((layer_output.shape[0], 50))))

    '''
	n_steps_in = 6
	n_steps_out = 3

    ''' 
    y = copy.copy(X_train)
    train_g = gen(X_train, layer_output, y, 50, lstm.vocab_size)

	# define model
    train, infenc, infdec = define_models(lstm.vocab_size, lstm.vocab_size, 128)	
    train.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

    print(X_train.shape,layer_output.shape,y.shape)

	# train model
	#train.fit([X1, X2], y, epochs=1)
    train.fit_generator(train_g.__getitem__(), steps_per_epoch= math.ceil(len(X_train) / 50), epochs=1)

	# evaluate LSTM
	# total, correct = 100, 0
	# for _ in range(total):
	# 	X1, X2, y = get_dataset(n_steps_in, n_steps_out, n_features, 1)
	# 	X1, X2, y = categorical(X1, X2, y, n_features)

	    
	# 	target = predict_sequence(infenc, infdec, X1, n_steps_out, n_features)
	# 	if np.array_equal(one_hot_decode(y[0]), one_hot_decode(target)):
	# 		correct += 1
	# print('Accuracy: %.2f%%' % (float(correct)/float(total)*100.0))
	# # spot check some examples
	# for _ in range(10):
	# 	X1, X2, y = get_dataset(n_steps_in, n_steps_out, n_features, 1)
	# 	X1, X2, y = categorical(X1, X2, y, n_features)
	# 	target = predict_sequence(infenc, infdec, X1, n_steps_out, n_features)
	# 	print('X=%s y=%s, yhat=%s' % (one_hot_decode(X1[0]), one_hot_decode(y[0]), one_hot_decode(target)))
