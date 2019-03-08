#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# python my_script.py --my-config config.txt  
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import LeakyReLU
from keras.layers import PReLU
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
import keras.utils as ku
from keras.utils.vis_utils import plot_model
from sklearn.model_selection import train_test_split
import math
import pickle
from DataGenerator import Generator as gen
import matplotlib.pyplot as plt

import configargparse


class lstmEncoder:
    def __init__(self, batch_size):
        self.args = self._parse_args()
        docs, labels = self.load_data()
        self.docs = docs
        self.labels = labels
        self.batch_size = batch_size
        self.num_classes = len(np.unique(labels))
        self.vocab_size = 0 # not assign yet


    def _parse_args(self):
        p = configargparse.ArgParser()
        p.add('-c', '--config',required=True, is_config_file=True, help='config file path')
        p.add('--embedding_path', required=True)
        p.add('--data_path', required=True)
        args = p.parse_args()
        return args
    
    def set_limitData(self, X_train, y_train, X_val, y_val, X_test, y_test, limit=None):
        X_train = X_train[:limit]
        y_train = y_train[:limit]
        X_val = X_val[:limit]
        y_val = y_val[:limit]
        X_test = X_test[:limit]
        y_test = y_test[:limit]  
        return X_train, y_train, X_val, y_val, X_test, y_test

    def load_data(self):
        ### load intput text
        # "/Users/apple/Desktop/q2_course/cs272/finalProject/glove.6B/glove.6B.100d.txt"
        
        print("LOAD_DATA...")
        corpus = pickle.load( open( self.args.data_path , "rb" ) )
        docs = []
        labels = []  
        
        for c in corpus:
            docs.append(c[0])
            labels.append(c[2])
        labels = np.array(labels)
        del corpus 
        return docs, labels
    
    def create_Emb(self):
        ### prepare tokenizer
        t = Tokenizer()
        t.fit_on_texts(self.docs)
        self.vocab_size = len(t.word_index) + 1

        print("Vocab size: "+str(self.vocab_size))
        
        ### integer encode the documents
        encoded_docs = t.texts_to_sequences(self.docs)

        ### split in random
        print("Shuffling...")
        X_train, X_test, y_train, y_test = train_test_split(encoded_docs, self.labels, test_size=0.1, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        

        ### DEBUG: set data length
        X_train, y_train, X_val, y_val, X_test, y_test = self.set_limitData( X_train, y_train, X_val, y_val, X_test, y_test, 300000)
        self.trainLen = len(X_train)
        
        
        print("X_train: "+str(len(X_train)))
        print("y_train: "+str(len(y_train)))
        print("X_val: "+str(len(X_val)))
        print("y_val: "+str(len(y_val)))
        print("X_test: "+str(len(X_test)))
        print("y_test: "+str(len(y_test)))
        

        ### pad train data
        # self.max_train_len = max(len(x) for x in X_train)
        self.max_train_len = 20
        print("max_train_len: "+str(self.max_train_len))
        X_train = pad_sequences(X_train, maxlen=self.max_train_len, padding='pre')
        y_train = ku.to_categorical(y_train, num_classes=self.num_classes)
        X_val = pad_sequences(X_val, maxlen=self.max_train_len, padding='pre')
        y_val = ku.to_categorical(y_val, num_classes=self.num_classes)
        
        
        ### pad test data
        # self.max_test_len = max(len(x) for x in X_test)
        # self.max_test_len = 1000
        self.max_test_len = self.max_train_len
        print("max_test_len: "+str(self.max_test_len))
        X_test = pad_sequences(X_test, maxlen=self.max_test_len, padding='pre')
        y_test = ku.to_categorical(y_test, num_classes=self.num_classes)


        ### load the whole embedding into memory
        embeddings_index = dict()
        f = open( self.args.embedding_path, encoding="utf-8")
        # "/Users/apple/Desktop/q2_course/cs272/finalProject/CS272-NLP-Project/data"
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()
        print('Loaded %s word vectors.' % len(embeddings_index))
        
        ### create a weight matrix for words in training docs
        embedding_matrix = np.zeros((self.vocab_size, 100))
        for word, i in t.word_index.items():
        	embedding_vector = embeddings_index.get(word)
        	if embedding_vector is not None:
        		embedding_matrix[i] = embedding_vector
                
        # train_g = gen(X_train, y_train, self.batch_size, self.num_classes)
        # val_g = gen(X_val, y_val, self.batch_size, self.num_classes)

        # print("X_train: "+str(X_train[:3]))
        # print("y_train: "+str(y_train[:3]))
        # print("X_test: "+str(X_test[:3]))
        # print("y_test: "+str(y_test[:3]))
        # print("Embedding matrix: "+str(embedding_matrix[:3]))
        
        # return train_g, val_g, X_test, y_test, embedding_matrix
        return X_train, y_train, X_val, y_val, X_test, y_test, embedding_matrix


    def buildModel(self, embedding_matrix):
        self.model = Sequential()
        # https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/
        self.model.add(Embedding(input_dim=self.vocab_size, output_dim=100, weights=[embedding_matrix], input_length=self.max_train_len))
        self.model.add(LSTM(50, go_backwards=True))
        self.model.add(PReLU())
        self.model.add(Dropout(rate=0.1)) 
        self.model.add(Dense(50, activation="selu"))
        self.model.add(Dropout(rate=0.1)) 
        self.model.add(Dense(50, activation="selu"))
        self.model.add(Dropout(rate=0.1)) 
        self.model.add(Dense(self.num_classes, activation='softmax'))
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
        print(self.model.summary())
        return self.model


        
        
    # def train(self,  train_g, val_g, X_test, y_test):
    def train(self, X_train, y_train, X_val, y_val, X_test, y_test):
        # self.model.fit_generator(train_g.__getitem__(), steps_per_epoch= math.ceil(self.trainLen / self.batch_size), epochs=50, 
        #                     validation_data=val_g.__getitem__(),validation_steps=50)

        history = self.model.fit(X_train, y_train, batch_size = self.batch_size, epochs = 1, shuffle=False, validation_data=(X_val, y_val))
        
        plt.plot(history.history['categorical_accuracy'])
        plt.plot(history.history['val_categorical_accuracy'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()
        plt.savefig('acc.png')
        
        # Plot training & validation loss values
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()
        plt.savefig('lose.png')


        #saves model
        try:
            file_name = "./classifier.h5"
            self.model.save(file_name)
        except Exception as error:
            print("Couldn't save model")
        
        ### evaluate the model
        #loss, accuracy = model.evaluate(padded_docs, labels, verbose=0)
        loss, accuracy = self.model.evaluate(X_test, y_test)
        print("loss %f " % (loss*100))
        print('Accuracy: %f' % (accuracy*100))



if __name__ == "__main__":     
    batch_size = 50
    lstm = lstmEncoder(batch_size)
    train_g, val_g, X_val, y_val,X_test, y_test, embedding_matrix = lstm.create_Emb()
    lstm.buildModel(embedding_matrix)
    lstm.train(train_g, val_g, X_val, y_val, X_test, y_test)
    plot_model(lstm.model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)