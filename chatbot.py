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


    #returns the state of the last hidden layer in model
    def get_hidden_layer_output(self, model, X_train):
        # print("Num layers: "+str(len(model.layers)))
        # print(model.layers)
        # input()

        # [<keras.layers.embeddings.Embedding object at 0x0000029185384D68>, 
        # <keras.layers.recurrent.LSTM object at 0x000002915FEAD9B0>, 
        # <keras.layers.advanced_activations.PReLU object at 0x00000291853A2A20>, 
        # <keras.layers.core.Dropout object at 0x00000291A74E1F28>, 
        # <keras.layers.core.Dense object at 0x00000291A74C12B0>, 
        # <keras.layers.core.Dropout object at 0x0000029185DC2A20>, 
        # <keras.layers.core.Dense object at 0x0000029185DC2C18>, 
        # <keras.layers.core.Dropout object at 0x0000029185DF35F8>, 
        # <keras.layers.core.Dense object at 0x0000029185DE6F28>]



        # get hidden layer output
        get_last_hidden_layer_output = K.function([model.layers[0].input],
                                      [model.layers[-2].output])

        # print(get_last_hidden_layer_output)
        # print()
        returned = get_last_hidden_layer_output([X_train])
        # print("Returned: "+str(returned))
        # input()
        layer_output = returned[0]
        # print("Layer output: "+str(layer_output))
        # print()
        # input()

        #results include positive and negative numbers, because the output is from 
        #the dense layer that's right before the last output layer, so right before the last activation function is applied


        
        #layer_output = np.hstack((layer_output,np.zeros((layer_output.shape[0], 50)))) 
        return layer_output



    # builds seq2seq model with encoder and decoder
    def build_model(self):
     #    word_dim = 100

     #    GRU_sizes = 25
        
     #    #initializes hidden layer initial state
     #    hiddenLayer_state_inputs = Input(shape=(GRU_sizes,))
     #    hiddenLayer_state = [hiddenLayer_state_inputs]

     #    # self.model.add(Embedding(input_dim=self.vocab_size, output_dim=100, weights=[embedding_matrix], input_length=self.max_train_len))

     #    word_vec_input = Input(shape=(self.max_train_length,))
     #    decoder_embed = Embedding(input_dim=self.encoder_decoder_handler.vocab_size, output_dim=100, weights=[self.embedding_matrix], input_length=self.max_train_length)
     #    embedded = decoder_embed(word_vec_input)


     #    decoder_gru_1 = GRU(GRU_sizes, return_sequences=True, return_state=False)
     #    decoder_gru_2 = GRU(GRU_sizes, return_sequences=True, return_state=True)
     #    decoder_dense = Dense(self.encoder_decoder_handler.vocab_size, activation='softmax') #size of the number of unique words
        
        

     #    gru_1_output = decoder_gru_1(embedded, initial_state=hiddenLayer_state)

     #    #gru_1 feeds into gru_2
     #    gru_2_output, state_h = decoder_gru_2(gru_1_output)
     #    decoder_outputs = decoder_dense(gru_2_output)
        
     #    # Define the model that will be used for training
     #    training_model = Model([word_vec_input, hiddenLayer_state_inputs], decoder_outputs)
     #    training_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
     #    print(training_model.summary())
        

    	# # Define inference decoder
     #    decoder_states_inputs = [Input(shape=(GRU_sizes,))]
        
     #    embedded_word = Input(shape=(1,100))
     #    gru_1_output =  decoder_gru_1(embedded_word, initial_state=decoder_states_inputs)
     #    gru_2_output, state_h = decoder_gru_2(gru_1_output)

     #    decoder_states = [state_h]
     #    decoder_outputs = decoder_dense(gru_2_output)
     #    inference_model = Model([embedded_word] + decoder_states_inputs, [decoder_outputs] + decoder_states)
     #    print(inference_model.summary())
     #    return training_model, inference_model


        latent_dim = 256

        # word_vec_input = Input(shape=(self.max_train_length,), name="input_1")
        # encoder_embed = Embedding(input_dim=self.encoder_decoder_handler.vocab_size, output_dim=100, weights=[self.embedding_matrix], input_length=self.max_train_length)
        # embedded = encoder_embed(word_vec_input)

        # # Define an input sequence and process it.
        # encoder_inputs = Input(shape=(None, self.encoder_decoder_handler.vocab_size), name="input_2")
        # encoder = LSTM(latent_dim, return_state=True)
        # encoder_outputs, state_h, state_c = encoder(encoder_inputs)
        # # We discard `encoder_outputs` and only keep the states.
        # encoder_states = [state_h, state_c]





        word_vec_input = Input(shape=(None,), name="input_1")
        encoder_embed = Embedding(input_dim=self.encoder_decoder_handler.vocab_size, output_dim=100, weights=[self.embedding_matrix], input_length=self.max_train_length)(word_vec_input)
        # Define an input sequence and process it.
        # encoder_inputs = Input(shape=(None, self.encoder_decoder_handler.vocab_size), name="input_2")
        # encoder = LSTM(latent_dim, return_state=True)(encoder_embed)
        encoder_outputs, state_h, state_c = LSTM(latent_dim, return_state=True)(encoder_embed)
        # We discard `encoder_outputs` and only keep the states.
        encoder_states = [state_h, state_c]






        # configure
        # num_encoder_tokens = 71
        # num_decoder_tokens = 93
        # latent_dim = 256

        # Set up the decoder, using `encoder_states` as initial state.
        decoder_inputs = Input(shape=(None, self.encoder_decoder_handler.vocab_size), name="input_3")
        decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
        decoder_dense = Dense(self.encoder_decoder_handler.vocab_size, activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)
        encoder_decoder = Model([word_vec_input, decoder_inputs], decoder_outputs)

        encoder_decoder.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['categorical_accuracy'])




        # # Set up the decoder, using `encoder_states` as initial state.
        # decoder_inputs = Input(shape=(None, self.encoder_decoder_handler.vocab_size), name="input_3")

        # decoder_embed = Embedding(input_dim=self.encoder_decoder_handler.vocab_size, output_dim=100, weights=[self.embedding_matrix])
        # final_embed = decoder_embed(decoder_inputs)

        # decoder_lstm = LSTM(latent_dim,  return_sequences=True, return_state=True)
        # decoder_outputs, _, _ = decoder_lstm(final_embed, initial_state=encoder_states)

        # #doesn't work
        # decoder_dense = Dense(self.encoder_decoder_handler.vocab_size, activation='softmax')
        # decoder_dense_outputs = decoder_dense(decoder_outputs)
        # encoder_decoder = Model([word_vec_input, decoder_inputs], decoder_dense_outputs)

        # # #does work
        # # decoder_dense = Dense(self.encoder_decoder_handler.vocab_size, activation='softmax')(decoder_outputs)
        # # encoder_decoder = Model([word_vec_input, decoder_inputs], decoder_dense)

        # encoder_decoder.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])




        # # Define an input sequence and process it.
        # encoder_inputs = Input(shape=(None,))
        # x = Embedding(num_encoder_tokens, latent_dim)(encoder_inputs)
        # x, state_h, state_c = LSTM(latent_dim,
        #                            return_state=True)(x)
        # encoder_states = [state_h, state_c]

        # # Set up the decoder, using `encoder_states` as initial state.
        # decoder_inputs = Input(shape=(None,))
        # x = Embedding(num_decoder_tokens, latent_dim)(decoder_inputs)
        # x = LSTM(latent_dim, return_sequences=True)(x, initial_state=encoder_states)
        # decoder_outputs = Dense(num_decoder_tokens, activation='softmax')(x)

        # # Define the model that will turn
        # # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
        # model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

        # # Compile & run training
        # model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
        # # Note that `decoder_target_data` needs to be one-hot encoded,
        # # rather than sequences of integers like `decoder_input_data`!
        # model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
        #           batch_size=batch_size,
        #           epochs=epochs,
        #           validation_split=0.2)



        # embedding_size = 50
        # from keras.layers import Input, LSTM, Embedding, Dense
        # from keras.models import Model
        # from keras.utils import plot_model
        # encoder_inputs = Input(shape=(None,))
        # # English words embedding
        # en_x=  Embedding(num_encoder_tokens, embedding_size)(encoder_inputs)
        # # Encoder lstm
        # encoder = LSTM(50, return_state=True)
        # encoder_outputs, state_h, state_c = encoder(en_x)
        # # We discard `encoder_outputs` and only keep the states.
        # encoder_states = [state_h, state_c]
        # # Set up the decoder, using `encoder_states` as initial state.
        # decoder_inputs = Input(shape=(None,))
        # # french word embeddings
        # dex=  Embedding(num_decoder_tokens, embedding_size)
        # final_dex= dex(decoder_inputs)
        # # decoder lstm
        # decoder_lstm = LSTM(50, return_sequences=True, return_state=True)
        # decoder_outputs, _, _ = decoder_lstm(final_dex,  initial_state=encoder_states)
        # decoder_dense = Dense(num_decoder_tokens, activation='softmax')
        # decoder_outputs = decoder_dense(decoder_outputs)
        # # While training, model takes eng and french words and outputs #translated french word
        # model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        # # rmsprop is preferred for nlp tasks
        # model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])




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


        # # define encoder inference model
        # inference_encoder_model = Model(word_vec_input, encoder_states)
        # # define decoder inference model
        # decoder_state_input_h = Input(shape=(latent_dim,))
        # decoder_state_input_c = Input(shape=(latent_dim,))
        # decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        # final_dex2= decoder_embed(decoder_inputs)
        # decoder_outputs, state_h, state_c = decoder_lstm(final_dex2, initial_state=decoder_states_inputs)

        # decoder_states = [state_h, state_c]
        # decoder_outputs = decoder_dense(decoder_outputs)
        # inference_decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs]+decoder_states)








        # # define the encoder model 
        # encoder_model = Model(word_vec_input, encoder_states)
        # encoder_model.summary()
        # # Redefine the decoder model with decoder will be getting below inputs from encoder while in prediction
        # decoder_state_input_h = Input(shape=(50,))
        # decoder_state_input_c = Input(shape=(50,))
        # decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        # final_dex2= dex(decoder_inputs)
        # decoder_outputs2, state_h2, state_c2 = decoder_lstm(final_dex2, initial_state=decoder_states_inputs)
        # decoder_states2 = [state_h2, state_c2]
        # decoder_outputs2 = decoder_dense(decoder_outputs2)
        # # sampling model will take encoder states and decoder_input(seed initially) and output the predictions(french word index) We dont care about decoder_states2
        # decoder_model = Model(
        #     [decoder_inputs] + decoder_states_inputs,
        #     [decoder_outputs2] + decoder_states2)



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

        # self.X_train = np.array(new_X_train)
        # self.y_train = np.array(new_y_train)
        # self.X_test = np.array(new_X_test)
        # self.y_test = np.array(new_y_test)

        self.decoder_X_train = new_X_train
        self.decoder_y_train = new_y_train
        self.decoder_X_test = new_X_test
        self.decoder_y_test = new_y_test

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test


        # self.X_train = np.reshape( chatbot.X_train, (chatbot.X_train.shape[0], self.max_train_length, self.encoder_decoder_handler.vocab_size))

        print("lengths of data: ")
        print("X_train: "+str(len(self.X_train)))
        print("y_train: "+str(len(self.y_train)))
        # print("X_val: "+str(len(self.X_val)))
        # print("y_val: "+str(len(self.y_val)))
        print("X_test: "+str(len(self.X_test)))
        print("y_test: "+str(len(self.y_test)))
        print("embedding matrix: "+str(len(self.embedding_matrix)))
        # input()
        print()

    def predict_sequence(self, inference_encoder_model, inference_decoder_model, input_sequence):
        # Reverse-lookup token index to decode sequences back to something readable.
        # reverse_input_char_index = dict((i, char) for char, i in input_token_index.items())
        # reverse_target_char_index = dict((i, char) for char, i in target_token_index.items())


        original_sequence = self.encoder_decoder_handler.decode_sequence(input_sequence[0])

        print("Input sequence: "+str(original_sequence))






        # Encode the input as state vectors.
        # print(input_sequence.shape)
        states_value = inference_encoder_model.predict(input_sequence)

        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1, self.encoder_decoder_handler.vocab_size))
        # Populate the first character of target sequence with the start character.
        target_seq[0, 0, self.encoder_decoder_handler.get_token_index("anime")] = 1.

        # print(target_seq[0][0])

        # print(input_sequence[0][1])
        # target_seq[0][0] = input_sequence[0][1]

        # print(target_seq[0][0])




        # print("Starting sequence: "+str(target_seq))

        # Sampling loop for a batch of sequences
        # (to simplify, here we assume a batch of size 1).
        stop_condition = False
        decoded_sentence = ""
        while not stop_condition:
            output_tokens, h, c = inference_decoder_model.predict([target_seq] + states_value)
            # print("output tokens: "+str(output_tokens))

            # Sample a token
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_char = self.encoder_decoder_handler.decode_int(sampled_token_index)

            # print("sampled word: "+str(sampled_char))

            decoded_sentence += sampled_char+" "

            # Exit condition: either hit max length
            # or find stop character.
            if (sampled_char == 'eos' or len(decoded_sentence) > 50):
                stop_condition = True

            # print(target_seq.shape)

            #Update the target sequence (of length 1).
            target_seq = np.array([np.concatenate((target_seq[0], np.zeros((1, self.encoder_decoder_handler.vocab_size))))])
            target_seq[0, -1, sampled_token_index] = 1.

            # target_seq = np.zeros((1, 1, self.encoder_decoder_handler.vocab_size))
            # target_seq[0, 0, sampled_token_index] = 1.

            # Update states
            states_value = [h, c]

            # input("Continue...")

        print("Decoded sequence: "+str(decoded_sentence))
        print()
        return decoded_sentence



        # input("Continue...")













if __name__ == "__main__":     
    batch_size = 50
    print("Loading training, validation, and text data")
    
    # #builds classifier model, and loads the data
    # lstm = lstmEncoder(batch_size)

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


    # lstm.model.load_weights("./classifier.h5")

    # train
    print("Training Generator")
    # train_g = Generator(X_train, layer_output, X_train.copy(), lstm.batch_size, lstm.vocab_size)
    # training_model.fit_generator(train_g.__getitem__(), steps_per_epoch= math.ceil(len(X_train) / lstm.batch_size), epochs=3)





    print(chatbot.X_train.shape)
    print(chatbot.y_train.shape)
    print(chatbot.X_train[0])

    # self.model.fit(X_train, y_train, batch_size = self.batch_size, epochs = 40, shuffle=False)
    training_model.fit([chatbot.X_train, chatbot.decoder_X_train], chatbot.decoder_y_train, batch_size=500, epochs=30, shuffle=False)


    # save model
    print("Saving model")
    try:
        # training_model.save(model_path)
        # inf_file_name = "./seq2seqInference.h5"
        # inference_model.save(inf_file_name)

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
    total, correct = 100, 0
    # test_layer_output = get_hidden_layer_output(lstm, X_test[:total])
    # y_t = copy.copy(X_test[:total])


    for x in range(0, 30):
        input_sequence = chatbot.X_train[x : x+1]
        chatbot.predict_sequence(inference_encoder_model, inference_decoder_model, input_sequence)


    
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
    
    