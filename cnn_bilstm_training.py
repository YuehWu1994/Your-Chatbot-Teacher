#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import keras
from keras.models import Model, load_model
from keras.layers import Dense, Input, Dropout, MaxPooling1D, Conv1D, GlobalMaxPool1D, Bidirectional
from keras.layers import LSTM, Lambda, Bidirectional, concatenate, BatchNormalization, Embedding
from keras.layers import TimeDistributed
from keras.optimizers import Adam
from keras.models import model_from_json
from sklearn import preprocessing
from sklearn.utils import class_weight
import tensorflow as tf
import keras.backend as K

import IPython
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

from sklearn.model_selection import train_test_split
import configargparse
import pickle as pkl
class CharCNN:  
    CHAR_DICT = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .!?:,\'%-\(\)/$|&;[]"'
    
    def __init__(self, max_len_of_sentence, max_num_of_setnence, verbose=10):
        self.args = self._parse_args()
        self.data_path = self.args.data_path
        self.max_len_of_sentence = max_len_of_sentence
        self.max_num_of_setnence = max_num_of_setnence
        self.verbose = verbose
        
        self.num_of_char = 0
        self.num_of_label = 0
        self.unknown_label = ''
        self.labels = []
        self.docs = []

    def _parse_args(self):
        p = configargparse.ArgParser()
        p.add('-c', '--config',required=True, is_config_file=True, help='config file path')
        p.add('--data_path', required=True)
        p.add('--embedding_path', required=False)
        args = p.parse_args()
        return args
    
    def load_data(self,size_limit=None):
        with open(self.data_path , "rb" ) as f:
            corpus = pkl.load(f)
            for c in corpus:
                self.docs.append(c[0])
                self.labels.append(c[1])
            # self.labels = np.array(labels)
        del corpus
        self.labels = self.labels[:size_limit]
        self.docs = self.docs[:size_limit]
        
        '''
        class_weights = class_weight.compute_class_weight('balanced',
                                                  np.unique(self.labels),
                                                  self.labels)
        
        le = preprocessing.LabelEncoder()
        le.fit(self.labels)
        
        self.class_weights_dict = dict(zip(le.transform(list(le.classes_)),
                          class_weights))
        
        '''
        
        
        print('data size=', len(self.docs))
    def build_char_dictionary(self, char_dict=None, unknown_label='UNK'):
        """
            Define possbile char set. Using "UNK" if character does not exist in this set
        """ 
        
        if char_dict is None:
            char_dict = self.CHAR_DICT
            
        self.unknown_label = unknown_label

        chars = []

        for c in char_dict:
            chars.append(c)

        chars = list(set(chars))
        
        chars.insert(0, unknown_label)

        self.num_of_char = len(chars)
        self.char_indices = dict((c, i) for i, c in enumerate(chars))
        self.indices_char = dict((i, c) for i, c in enumerate(chars))
        
        if self.verbose > 5:
            print('Totoal number of chars:', self.num_of_char)

            print('First 3 char_indices sample:', {k: self.char_indices[k] for k in list(self.char_indices)[:3]})
            print('First 3 indices_char sample:', {k: self.indices_char[k] for k in list(self.indices_char)[:3]})
            

        return self.char_indices, self.indices_char, self.num_of_char
    
    def convert_labels(self):
        """
            Convert label to numeric
        """
        unq_labels = np.unique(np.array(self.labels))
        self.label2indexes = dict((l, i) for i, l in enumerate(unq_labels))
        self.index2labels = dict((i, l) for i, l in enumerate(unq_labels))

        if self.verbose > 5:
            print('Label to Index: ', self.label2indexes)
            print('Index to Label: ', self.index2labels)
            
        self.num_of_label = len(self.label2indexes)
        print("Number of class: ", self.num_of_label)

        return self.label2indexes, self.index2labels
    
    def _transform_raw_data(self, df, x_col, y_col, label2indexes=None, sample_size=None):
        """
            ##### Transform raw data to list
        """
        
        x = []
        y = []

        actual_max_sentence = 0
        
        if sample_size is None:
            sample_size = len(df)

        for i, row in df.head(sample_size).iterrows():
            x_data = row[x_col]
            y_data = row[y_col]

            sentences = sent_tokenize(x_data)
            x.append(sentences)

            if len(sentences) > actual_max_sentence:
                actual_max_sentence = len(sentences)

            y.append(label2indexes[y_data])

        if self.verbose > 5:
            print('Number of news: %d' % (len(x)))
            print('Actual max sentence: %d' % actual_max_sentence)

        return x, y
    
    def _transform_training_data(self, x_raw, y_raw, max_len_of_sentence=None, max_num_of_setnence=None):
        """
            ##### Transform preorcessed data to numpy
        """
        #import pdb; pdb.set_trace()
        unknown_value = self.char_indices[self.unknown_label]
        
        x = np.ones((len(x_raw), max_num_of_setnence, max_len_of_sentence), dtype=np.int64) * unknown_value
        y = np.array(y_raw)
        
        if max_len_of_sentence is None:
            max_len_of_sentence = self.max_len_of_sentence
        if max_num_of_setnence is None:
            max_num_of_setnence = self.max_num_of_setnence

        
        '''
        for i, doc in enumerate(x_raw):
            for j, sentence in enumerate(doc):
                if j < max_num_of_setnence:
                    for t, char in enumerate(sentence[-max_len_of_sentence:]):
                        if char not in self.char_indices:
                            x[i, j, (max_len_of_sentence-1-t)] = self.char_indices['UNK']
                        else:
                            x[i, j, (max_len_of_sentence-1-t)] = self.char_indices[char]
        '''                    
        for i, doc in enumerate(x_raw):
            for j in range(max_num_of_setnence):
                for k in range(max_len_of_sentence):
                    if j*max_len_of_sentence+k >= len(doc):
                        break
                    char = doc[j*max_len_of_sentence+k]
                    if char not in self.char_indices:
                        x[i, j, k] = self.char_indices['UNK']
                    else:
                        x[i, j, k] = self.char_indices[char]  
        '''
        for i, doc in enumerate(x_raw):
            for t, char in enumerate(doc[-max_len_of_sentence:]):
                if char not in self.char_indices:
                    x[i, 0, (max_len_of_sentence-1-t)] = self.char_indices['UNK']
                else:
                    x[i, 0, (max_len_of_sentence-1-t)] = self.char_indices[char]
        '''
        return x, y

    def _build_character_block(self, block, dropout=0.3, filters=[64, 100], kernel_size=[3, 3], 
                         pool_size=[2, 2], padding='valid', activation='relu', 
                         kernel_initializer='glorot_normal'):
        
        for i in range(len(filters)):
            block = Conv1D(
                filters=filters[i], kernel_size=kernel_size[i],
                padding=padding, activation=activation, kernel_initializer=kernel_initializer)(block)

        block = Dropout(dropout)(block)
        block = MaxPooling1D(pool_size=pool_size[i])(block)

        block = GlobalMaxPool1D()(block)
        block = Dense(128, activation='relu')(block)
        return block
    
    def _build_sentence_block(self, max_len_of_sentence, max_num_of_setnence, 
                              char_dimension=16,
                              filters=[[3, 5, 7], [200, 300, 300], [300, 400, 400]], 
#                               filters=[[100, 200, 200], [200, 300, 300], [300, 400, 400]], 
                              kernel_sizes=[[4, 3, 3], [5, 3, 3], [6, 3, 3]], 
                              pool_sizes=[[2, 2, 2], [2, 2, 2], [2, 2, 2]],
                              dropout=0.4):
        
        sent_input = Input(shape=(max_len_of_sentence, ), dtype='int64')
        embedded = Embedding(self.num_of_char, char_dimension, input_length=max_len_of_sentence)(sent_input)
        
        blocks = []
        for i, filter_layers in enumerate(filters):
            blocks.append(
                self._build_character_block(
                    block=embedded, filters=filters[i], kernel_size=kernel_sizes[i], pool_size=pool_sizes[i])
            )

        sent_output = concatenate(blocks, axis=-1)
        sent_output = Dropout(dropout)(sent_output)
        sent_encoder = Model(inputs=sent_input, outputs=sent_output)

        return sent_encoder
    
    def _build_document_block(self, sent_encoder, max_len_of_sentence, max_num_of_setnence, 
                             num_of_label, dropout=0.3, 
                             loss='sparse_categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy']):
        doc_input = Input(shape=(max_num_of_setnence, max_len_of_sentence), dtype='int64')
        doc_output = TimeDistributed(sent_encoder)(doc_input)


        #doc_output = Bidirectional(LSTM(128, return_sequences=True, dropout=dropout, recurrent_dropout=dropout, activation='tanh'))(doc_output)
        #doc_output = Dropout(dropout)(doc_output)        
        doc_output = Bidirectional(LSTM(128, return_sequences=False, dropout=dropout, recurrent_dropout=dropout, activation='tanh'))(doc_output)
        doc_output = Dropout(dropout)(doc_output)
        
        doc_output = Dense(128, activation='relu')(doc_output)
        doc_output = Dropout(dropout)(doc_output)
        doc_output = Dense(num_of_label, activation='softmax')(doc_output)

        doc_encoder = Model(inputs=doc_input, outputs=doc_output)
        doc_encoder.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        return doc_encoder
    
    def preprocess(self, char_dict=None, unknown_label='UNK'):
        if self.verbose > 3:
            print('-----> Stage: preprocess')
            
        self.build_char_dictionary(char_dict, unknown_label)
        self.convert_labels()
    
    def process(self, x, y,
                max_len_of_sentence=None, max_num_of_setnence=None, label2indexes=None, sample_size=None):
        if self.verbose > 3:
            print('-----> Stage: process')
            
        if sample_size is None:
            sample_size = 1000
        if label2indexes is None:
            if self.label2indexes is None:
                raise Exception('Does not initalize label2indexes. Please invoke preprocess step first')
            label2indexes = self.label2indexes
        if max_len_of_sentence is None:
            max_len_of_sentence = self.max_len_of_sentence
        if max_num_of_setnence is None:
            max_num_of_setnence = self.max_num_of_setnence

        # x_preprocess, y_preprocess = self._transform_raw_data(
        #     x,y, label2indexes=label2indexes)
        
        x_preprocess, y_preprocess = self._transform_training_data(
            x_raw=x, y_raw=y,
            max_len_of_sentence=max_len_of_sentence, max_num_of_setnence=max_num_of_setnence)
        
        if self.verbose > 5:
            print('Shape: ', x_preprocess.shape, y_preprocess.shape)

        return x_preprocess, y_preprocess
    
    def build_model(self, char_dimension=16, display_summary=False, display_architecture=False, 
                    loss='sparse_categorical_crossentropy', optimizer='rmsprop', metrics=['categorical_accuracy']):
        if self.verbose > 3:
            print('-----> Stage: build model')
            
        sent_encoder = self._build_sentence_block(
            char_dimension=char_dimension,
            max_len_of_sentence=self.max_len_of_sentence, max_num_of_setnence=self.max_num_of_setnence)
                
        doc_encoder = self._build_document_block(
            sent_encoder=sent_encoder, num_of_label=self.num_of_label,
            max_len_of_sentence=self.max_len_of_sentence, max_num_of_setnence=self.max_num_of_setnence, 
            loss=loss, optimizer=optimizer, metrics=metrics)
        
        if display_architecture:
            print('Sentence Architecture')
            IPython.display.display(SVG(model_to_dot(sent_encoder).create(prog='dot', format='svg')))
            print()
            print('Document Architecture')
            IPython.display.display(SVG(model_to_dot(doc_encoder).create(prog='dot', format='svg')))
        
        if display_summary:
            print(doc_encoder.summary())
            
        
        self.model = {
            'sent_encoder': sent_encoder,
            'doc_encoder': doc_encoder
        }
        
        return doc_encoder
    
    def train(self, x_train, y_train, x_test, y_test, batch_size=128, epochs=1, shuffle=True):
        if self.verbose > 3:
            print('-----> Stage: train model')
            
        self.get_model().fit(
            x_train, y_train, validation_data=(x_test, y_test), 
            batch_size=batch_size, epochs=epochs, shuffle=shuffle)
        
#         return self.model['doc_encoder']

    def predict(self, x, return_prob=False):
        if self.verbose > 3:
            print('-----> Stage: predict')
            
        if return_prob:
            return self.get_model().predict(x)
        
        return self.get_model().predict(x).argmax(axis=-1)
    
    def get_model(self):
        return self.model['doc_encoder']
    
    def save_model(self):
        # serialize model to JSON
        model_json = self.model.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights("model.h5")
        print("Saved model to disk")
        
    def load_model(self):
        # load json and create model
        json_file = open('model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)
        # load weights into new model
        self.model.load_weights("model.h5")
        
        print("Loaded model from disk")

if __name__ == '__main__':
    """
    Maximum number of characters per sentence is 256.
    Maximum number of sentence is 5
    """

    char_cnn = CharCNN(max_len_of_sentence=256, max_num_of_setnence=1)

    """
    First of all, we need to prepare meta information including character dictionary 
    and converting label from text to numeric (as keras support numeric input only).
    """
    
    """
    We have to transform raw input training data and testing to numpy format for keras input
    """
    char_cnn.load_data(10000)
    X_train, X_test, y_train, y_test = train_test_split(char_cnn.docs, char_cnn.labels, test_size=0.4, random_state=42)
    # X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    char_cnn.preprocess()
    x_train, y_train = char_cnn.process(X_train, y_train)
    x_test, y_test = char_cnn.process(X_test, y_test)

    char_cnn.build_model()
    char_cnn.train(x_train, y_train, x_test, y_test, batch_size=64, epochs=8)
    
    p = char_cnn.predict(x_test, False)
    for i in range(200):
        print(p[i])
        
    char_cnn.save_model()
    
    
    
