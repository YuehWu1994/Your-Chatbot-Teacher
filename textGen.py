#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from keras import backend as K
from keras.layers import Dense
from keras.models import Model
from keras.layers import Embedding
from keras.layers import GRU, Input
from lstmEnc_DNN import lstmEncoder 




if __name__ == "__main__":     
    batch_size = 50
    lstm = lstmEncoder(batch_size)
    X_train, y_train, X_val, y_val, X_test, y_test, embedding_matrix = lstm.create_Emb()
    lstm.buildModel(embedding_matrix)
    lstm.model.load_weights("./classifier.h5")
    
    print("get last hidden layer output")
    get_last_hidden_layer_output = K.function([lstm.model.layers[0].input],
                                  [lstm.model.layers[-2].output])
    layer_output = get_last_hidden_layer_output([X_train])[0]
    
    
    word_dim = 100
    
    word_vec_input = Input(shape=(word_dim,))
    hiddenLayer_inputs = Input(shape=(50,))
    decoder_embed = Embedding(input_dim=lstm.vocab_size, output_dim=word_dim, weights=[embedding_matrix], input_length=None, trainable=False)
    decoder_gru_1 = GRU(word_dim, return_sequences=True, return_state=False)
    decoder_gru_2 = GRU(word_dim, return_sequences=True, return_state=True)
    decoder_dense = Dense(lstm.vocab_size, activation='softmax')
    
    # Connect the layers
    embedded = decoder_embed(word_vec_input)
    gru_1_output = decoder_gru_1(embedded, initial_state=hiddenLayer_inputs)
    gru_2_output, state_h = decoder_gru_2(gru_1_output)
    decoder_outputs = decoder_dense(gru_2_output)
    
    # Define the model that will be used for training
    training_model = Model([word_vec_input, hiddenLayer_inputs], decoder_outputs)
    
    # Also create a model for inference (this returns the GRU state)
    decoder_model = Model([word_vec_input, hiddenLayer_inputs], [decoder_outputs, state_h])
    
    
    training_model.fit([X_train, layer_output], X_train, epochs=1);
    