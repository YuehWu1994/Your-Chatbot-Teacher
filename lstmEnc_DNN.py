#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# python my_script.py --my-config config.txt  
import numpy as np
import os
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
import itertools
import pickle as pkl
from DataGenerator import Generator as gen
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import configargparse


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


class lstmEncoder:
    def __init__(self, batch_size):
        #self.args = self._parse_args()
        self.batch_size = batch_size
        
        cwdFiles = os.listdir(os.getcwd())
        if('enc_doc.pkl' not in cwdFiles or 'label.pkl' not in cwdFiles or 'word_index.pkl' not in cwdFiles):  
            self.load_data()
            
        self.encoded_docs = pkl.load( open('./enc_doc.pkl', "rb" ) )
        self.labels = pkl.load(open('./label.pkl', "rb" ))
        self.word_index = pkl.load(open('./word_index.pkl', "rb" ))
        self.index_word = dict()
        for i in self.word_index:
            self.index_word[self.word_index[i]] = i
        self.num_classes = len(np.unique(self.labels))
        self.vocab_size = len(self.word_index) + 1
        
        print("Number of class: ", self.num_classes)


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
        # "/Users/apple/Desktop/q2_course/cs272/finalProject/CS272-NLP-Project/data"  self.args.data_path
        print("LOAD_DATA...")

        corpus = pkl.load( open("/Users/apple/Desktop/q2_course/cs272/finalProject/CS272-NLP-Project/data" , "rb" ) )
        docs = []
        labels = []  
        
        for c in corpus:
            docs.append(c[0])
            labels.append(c[2]) # c[1] for meta class(7 classes)  / c[2] for meta class(51 classes) 
        labels = np.array(labels)

        del corpus 
        
        print("Tokenize...")     
        t = Tokenizer()
        t.fit_on_texts(docs)
        ### integer encode the documents
        encoded_docs = t.texts_to_sequences(docs)
        
        print("WRITE PKL")
        with open('./enc_doc.pkl','wb') as f:
            pkl.dump(encoded_docs,f)
        with open('./label.pkl','wb') as f:
            pkl.dump(labels,f)
        with open('./word_index.pkl','wb') as f:
            pkl.dump(t.word_index,f)  
    
    def create_Emb(self, limitNum):
        ### split in random
        print("Shuffling...")
        X_train, X_test, y_train, y_test = train_test_split(self.encoded_docs, self.labels, test_size=0.1, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        

        ### DEBUG: set data length
        X_train, y_train, X_val, y_val, X_test, y_test = self.set_limitData( X_train, y_train, X_val, y_val, X_test, y_test, limitNum)
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
        f = open( "/Users/apple/Desktop/q2_course/cs272/finalProject/glove.6B/glove.6B.100d.txt" , encoding="utf-8")
        # self.args.embedding_path  "/Users/apple/Desktop/q2_course/cs272/finalProject/glove.6B/glove.6B.100d.txt"
        
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()
        print('Loaded %s word vectors.' % len(embeddings_index))
        
        ### create a weight matrix for words in training docs                
        embedding_matrix = np.zeros((self.vocab_size, 100))
        for word, i in self.word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

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

        self.history = self.model.fit(X_train, y_train, batch_size = self.batch_size, epochs = 18, 
                                 shuffle=False, validation_data=(X_val, y_val))
        fig = plt.figure()
        plt.plot(self.history.history['categorical_accuracy'], 'b')
        plt.plot(self.history.history['val_categorical_accuracy'], 'g')
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()
        fig.savefig('acc.png')
        
        # Plot training & validation loss values
        fig = plt.figure()
        plt.plot(self.history.history['loss'], 'b')
        plt.plot(self.history.history['val_loss'], 'g')
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()
        fig.savefig('loss.png')


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
    train_g, val_g, X_val, y_val,X_test, y_test, embedding_matrix = lstm.create_Emb(300000)
    lstm.buildModel(embedding_matrix)
    lstm.model.load_weights("/Users/apple/Desktop/q2_course/cs272/finalProject/CS272-NLP-Project/classifier.h5")
    cnf_matrix=confusion_matrix(np.argmax(y_test, axis = 1), np.argmax(lstm.model.predict(X_test), axis = 1))
np.set_printoptions(precision=2)
plt.figure(figsize=(20,10))

#plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,title='Normalized confusion matrix class weight unbalanced')
    #lstm.train(train_g, val_g, X_val, y_val, X_test, y_test)
    #plot_model(lstm.model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)