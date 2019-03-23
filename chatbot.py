#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#just for James
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Dense, TimeDistributed
from keras.models import Model
from keras.layers import Embedding
from keras.layers import GRU, Input
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Flatten
from keras.models import model_from_json
# from lstmEnc_DNN import lstmEncoder 
from encoder_decoder_handler import encoder_decoder_handler
from evaluate import countBLEU
import keras.utils as ku
import copy
import math


class Chatbot:


    #hyperparameters
    batch_size = 0

    #data loading methods
    encoder_decoder_handler = None

    #classifier model object
    classifier_model = None
    max_train_length = 20

    #data
    train_X = []
    X_train = []
    y_train = []
    X_val = []
    y_val = []
    X_test = []
    y_test = []
    embedding_matrix = []


    def __init__(self):
        self.batch_size = 50

        self.encoder_decoder_handler = encoder_decoder_handler(self.batch_size)


    # builds seq2seq model with encoder and decoder
    def build_model(self):
        latent_dim = 256


        #initializes encoder
        word_vec_input = Input(shape=(None,), name="input_1")
        encoder_embed = Embedding(input_dim=self.encoder_decoder_handler.vocab_size, output_dim=100, weights=[self.embedding_matrix], input_length=self.max_train_length)(word_vec_input)
        encoder_outputs, state_h, state_c = LSTM(latent_dim, return_state=True)(encoder_embed)
        encoder_states = [state_h, state_c]

        #initializes decoder
        decoder_inputs = Input(shape=(None, self.encoder_decoder_handler.vocab_size), name="input_3")
        decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
        decoder_dense = Dense(self.encoder_decoder_handler.vocab_size, activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)
        encoder_decoder = Model([word_vec_input, decoder_inputs], decoder_outputs)


        #compiles model
        encoder_decoder.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['categorical_accuracy'])



        # define encoder inference model
        inference_encoder_model = Model(word_vec_input, encoder_states)


        # define decoder inference model
        decoder_state_input_h = Input(shape=(latent_dim,))
        decoder_state_input_c = Input(shape=(latent_dim,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = decoder_dense(decoder_outputs)
        inference_decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs]+decoder_states)


        print(encoder_decoder.summary())

        return encoder_decoder, inference_encoder_model, inference_decoder_model


    #returns an inference model that can be used for predicting one word at a time
    def build_inference_model(self, encoder_decoder_model):
        #Defining the inference models requires reference to elements of the model used for training in the example. 
        #Alternately, one could define a new model with the same shapes and load the weights from file.

        pass



    def predict_sequence(self, inference_model, X1, X2, n_steps, cardinality):
        #encode
        X2 = np.reshape(X2, (1, len(X2)))
        state = [X2]   

        output = list()
        for t in range(n_steps):
            
            # predict next word
            x = np.reshape(embedding_matrix[X1[t]], (1,1,100))
            yhat, h = inference_model.predict([x] + state)
            # store prediction
            output.append(yhat[0,0,:])
            # update state
            state = [h]
            # update target sequence by next word
            #target_seq = X1[t+1]
        return np.array(output)

    def one_hot_decode(self, encoded_seq):
    	return [np.argmax(vector) for vector in encoded_seq]



    def interpret(self, lstm, y, target):
        ans = ""
        predSeq = ""
        for i in range(lstm.max_train_len):
            ans +=  "<UNK> " if (y[i] == 0) else (lstm.index_word[y[i]] + ' ')
            predSeq +=  "<UNK> " if (target[i] == 0) else (lstm.index_word[target[i]] + ' ')
        print(ans)
        print(predSeq)



    #loads data from the lstm_encoder class
    def load_data(self, num_data_points):

        X_train, y_train, X_test, y_test, self.embedding_matrix = self.encoder_decoder_handler.create_Emb(num_data_points)

        #3D vectorized representation of words
        new_X_train = np.zeros((len(X_train), self.max_train_length, self.encoder_decoder_handler.vocab_size), dtype='int8')
        new_y_train = np.zeros((len(y_train), self.max_train_length, self.encoder_decoder_handler.vocab_size), dtype='int8')
        new_X_test = np.zeros((len(X_test), self.max_train_length, self.encoder_decoder_handler.vocab_size), dtype='int8')
        new_y_test = np.zeros((len(y_test), self.max_train_length, self.encoder_decoder_handler.vocab_size), dtype='int8')


        ### converts [5, 12, 1, 482, 8] into a list of binary lengths with 1s at specific indices ###
        #converts training data
        for x in range(0, len(X_train)):
            for y in range(0, len(X_train[x])):
                x_train_index = X_train[x][y]
                y_train_index = y_train[x][y]

                new_X_train[x][y][x_train_index] = 1
                new_y_train[x][y][y_train_index] = 1

        #converts testing data
        for x in range(0, len(X_test)):
            for y in range(0, len(X_test[x])):
                x_test_index = X_test[x][y]
                y_test_index = y_test[x][y]

                new_X_test[x][y][x_test_index] = 1
                new_y_test[x][y][y_test_index] = 1
        self.decoder_X_train = new_X_train
        self.decoder_y_train = new_y_train
        self.decoder_X_test = new_X_test
        self.decoder_y_test = new_y_test

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test



        print("lengths of data: ")
        print("X_train: "+str(len(self.X_train)))
        print("y_train: "+str(len(self.y_train)))
        # print("X_val: "+str(len(self.X_val)))
        # print("y_val: "+str(len(self.y_val)))
        print("X_test: "+str(len(self.X_test)))
        print("y_test: "+str(len(self.y_test)))
        print("embedding matrix: "+str(len(self.embedding_matrix)))
        print()

    def predict_sequence(self, inference_encoder_model, inference_decoder_model, input_sequence):

        #converts hot one vector into a string sentence
        original_sequence = self.encoder_decoder_handler.decode_sequence(input_sequence[0])

        print("Input sequence: "+str(original_sequence))


        # Encode the input as state vectors.
        states_value = inference_encoder_model.predict(input_sequence)

        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1, self.encoder_decoder_handler.vocab_size))
        # Populate the first character of target sequence with the start character.
        target_seq[0, 0, self.encoder_decoder_handler.get_token_index("anime")] = 1.



        # Sampling loop for a batch of sequences
        # (to simplify, here we assume a batch of size 1).
        stop_condition = False
        decoded_sentence = ""
        while not stop_condition:
            output_tokens, h, c = inference_decoder_model.predict([target_seq] + states_value)

            # Sample a token
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_char = self.encoder_decoder_handler.decode_int(sampled_token_index)

            decoded_sentence += sampled_char+" "

            # Exit condition: either hit max length
            # or find stop character.
            if (sampled_char == 'eos' or len(decoded_sentence) > 50):
                stop_condition = True

            #Update the target sequence (of length 1).
            target_seq = np.array([np.concatenate((target_seq[0], np.zeros((1, self.encoder_decoder_handler.vocab_size))))])
            target_seq[0, -1, sampled_token_index] = 1.

            #resets the target sequence each time
            # target_seq = np.zeros((1, 1, self.encoder_decoder_handler.vocab_size))
            # target_seq[0, 0, sampled_token_index] = 1.

            # Update states
            states_value = [h, c]



        print("Decoded sequence: "+str(decoded_sentence))
        print()
        return decoded_sentence













if __name__ == "__main__": 
    print("Loading training, validation, and text data")
    
    chatbot = Chatbot()
    
    num_data_points = 20000
    chatbot.load_data(num_data_points)


    #if a model hasn't been trained yet, train one
    model_arch_path = "./seq2seq_arch.json"
    model_weight_path = "./seq2seq_weights.h5"
    inference_encoder_arch_path = "./seq2seq_inference_encoder_arch.json"
    inference_encoder_weight_path = "./seq2seq_inference_encoder_weights.h5"
    inference_decoder_arch_path = "./seq2seq_inference_decoder_arch.json"
    inference_decoder_weight_path = "./seq2seq_inference_decoder_weights.h5"

    
    # if os.path.isfile(model_arch_path)==False:
    
    
    #builds classifier model
    print("Building Encoder and Decoder model")
    training_model, inference_encoder_model, inference_decoder_model = chatbot.build_model()


    print("X_train shape: "+str(chatbot.X_train.shape))
    print("y_train shape: "+str(chatbot.y_train.shape))
    print(chatbot.X_train[0])

    #trains the model
    training_model.fit([chatbot.X_train, chatbot.decoder_X_train], chatbot.decoder_y_train, batch_size=500, epochs=60, shuffle=False)


    # save model
    print("Saving model")
    try:

        ## Saves training model ##
        # serialize model to JSON
        model_json = training_model.to_json()
        with open(model_arch_path, "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        training_model.save_weights(model_weight_path)


        ## Saves inference encoder model ##
        # serialize model to JSON
        inference_encoder = inference_encoder_model.to_json()
        with open(inference_encoder_arch_path, "w") as json_file:
            json_file.write(inference_encoder)
        # serialize weights to HDF5
        inference_encoder_model.save_weights(inference_encoder_weight_path)


        ## Saves inference decoder model ##
        # serialize model to JSON
        inference_decoder = inference_decoder_model.to_json()
        with open(inference_decoder_arch_path, "w") as json_file:
            json_file.write(inference_decoder)
        # serialize weights to HDF5
        inference_decoder_model.save_weights(inference_decoder_weight_path)




        print("Saved model to disk")
    except Exception as error:
        print("Couldn't save model: "+str(error))
    # else:
    #     # training_model = load_model(model_path)

    #     # load json and create model
    #     json_file = open(model_arch_path, 'r')
    #     loaded_model_json = json_file.read()
    #     json_file.close()
    #     training_model = model_from_json(loaded_model_json)
    #     # load weights into new model
    #     training_model.load_weights(model_weight_path)
    #     print("Loaded model from disk")

    # evaluate LSTM
    print("Evaluating results")


    print("Train data")
    for x in range(0, 50):
        input_sequence = chatbot.X_train[x : x+1]
        chatbot.predict_sequence(inference_encoder_model, inference_decoder_model, input_sequence)

    print("Test data")
    for x in range(0, 50):
        input_sequence = chatbot.X_test[x : x+1]
        chatbot.predict_sequence(inference_encoder_model, inference_decoder_model, input_sequence)



    
    total, correct = 100, 0
    # bleu = countBLEU(lstm)
    
    # for i in range(total):
    #     # extract indexes for this batch
    #     X1 = X_test[i]
    #     X2 = test_layer_output[i]
        
    #     y = ku.to_categorical([y_t[i]], num_classes=lstm.vocab_size)
    #     y = np.reshape(y, (y.shape[1], y.shape[2]))       
    #     target = predict_sequence(training_model, X1, X2, lstm.max_train_len, lstm.vocab_size)
        
    #     #print(one_hot_decode(y))
    #     #print(one_hot_decode(target))
    #     interpret(lstm, one_hot_decode(y), one_hot_decode(target))
    #     bleu.count_BLEU(one_hot_decode(y), one_hot_decode(target))
        
    #     if np.array_equal(one_hot_decode(y), one_hot_decode(target)):
    #         correct += 1
    # print('Accuracy: %.2f%%' % (float(correct)/float(total)*100.0))
    
    