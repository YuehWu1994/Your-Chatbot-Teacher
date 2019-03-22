# python my_script.py --config config/config.txt  
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
from sklearn.model_selection import train_test_split
import math
import pickle
from DataGenerator import Generator as gen

import configargparse


class encoder_decoder_handler:

    input_data = []
    target_data = []

    tokenizer = None

    def __init__(self, batch_size):
        self.args = self._parse_args()
        self.input_data, self.target_data = self.load_data()

        self.batch_size = batch_size
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
        corpus = pickle.load( open( "./encoder_decoder_data", "rb" ) )
        input_data = []
        target_data = []  
        
        for index, c in enumerate(corpus):
            if index<20000:
                input_data.append(c[0])
                target_data.append(c[1])

        print("Input Len:"+str(len(input_data)))
        print("Target Len: "+str(len(target_data)))

        # target_data = np.array(target_data)

        return input_data, target_data
    
    def create_Emb(self, num_data_points):

        #inserts <SOS> and <EOS> onto input and target sentences
        for x in range(0, len(self.input_data)):
            self.input_data[x] = "<SOS>"+self.input_data[x]+"<EOS>"

        for x in range(0, len(self.target_data)):
            self.target_data[x] = "<SOS>"+self.target_data[x]+"<EOS>"



        #prepare tokenizer
        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts(self.input_data)

        # t.fit_on_texts(self.target_data) #don't need to do this, because target data is just like input data, but offset by 1. 

        self.vocab_size = len(self.tokenizer.word_index) + 1

        print("Vocab size: "+str(self.vocab_size))
        
        ### integer encode the documents
        encoded_input = self.tokenizer.texts_to_sequences(self.input_data)
        encoded_target = self.tokenizer.texts_to_sequences(self.target_data)


        #only keeps the top 


        ### split in random
        print("Shuffling...")
        X_train, X_test, y_train, y_test = train_test_split(encoded_input, encoded_target, test_size=0.2, random_state=42)
        # X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        
        print("X_train: "+str(len(X_train)))
        print("y_train: "+str(len(y_train)))
        # print("X_val: "+str(len(X_val)))
        # print("y_val: "+str(len(y_val)))
        print("X_test: "+str(len(X_test)))
        print("y_test: "+str(len(y_test)))


        ### DEBUG: set data length
        # X_train, y_train, X_val, y_val, X_test, y_test = self.set_limitData( X_train, y_train, X_val, y_val, X_test, y_test, 100000)
        X_train = X_train[:num_data_points]
        y_train = y_train[:num_data_points]
        X_test = X_test[:num_data_points]
        y_test = y_test[:num_data_points]
        # X_test = X_test[:40000]
        # y_test = y_test[:40000]
        self.trainLen = len(X_train)

        

        ### pad train data
        # self.max_train_len = max(len(x) for x in X_train)
        self.max_train_len = 20
        print("max_train_len: "+str(self.max_train_len))
        X_train = pad_sequences(X_train, maxlen=self.max_train_len, padding='post')
        y_train = pad_sequences(y_train, maxlen=self.max_train_len, padding='post')
        # y_train = ku.to_categorical(y_train, num_classes=self.num_classes)

        
        
        ### pad test data
        # self.max_test_len = max(len(x) for x in X_test)
        # self.max_test_len = 1000
        self.max_test_len = self.max_train_len
        print("max_test_len: "+str(self.max_test_len))
        X_test = pad_sequences(X_test, maxlen=self.max_test_len, padding='post')
        y_test = pad_sequences(y_test, maxlen=self.max_test_len, padding='post')
        # y_test = ku.to_categorical(y_test, num_classes=self.num_classes)


        ### load the whole embedding into memory
        embeddings_index = dict()
        f = open(self.args.embedding_path, encoding="utf-8")
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float16')
            embeddings_index[word] = coefs
        f.close()
        print('Loaded %s word vectors.' % len(embeddings_index))
        
        ### create a weight matrix for words in training docs
        embedding_matrix = np.zeros((self.vocab_size, 100))
        for word, i in self.tokenizer.word_index.items():
        	embedding_vector = embeddings_index.get(word)
        	if embedding_vector is not None:
        		embedding_matrix[i] = embedding_vector
                
        # train_g = gen(X_train, y_train, self.batch_size, self.num_classes)
        # val_g = gen(X_val, y_val, self.batch_size, self.num_classes)

        print("X_train: "+str(X_train[:3]))
        print("y_train: "+str(y_train[:3]))
        print("X_test: "+str(X_test[:3]))
        print("y_test: "+str(y_test[:3]))
        # print("Embedding matrix: "+str(embedding_matrix[:3]))
        
        # return train_g, val_g, X_test, y_test, embedding_matrix
        return X_train, y_train, X_test, y_test, embedding_matrix

    #decodes sequence from hot one vectors to words
    def decode_sequence(self, sequence):

        #sequence is a 2D matrix with each index being a word, and each word being a binary vector with index of 1 representing its unique ID


        #sequence is a list of ints, corresponding to a word



        to_return = ""
        for x in range(0, len(sequence)):
            # index = np.where(sequence[x] == 1)[0][0]
            index = sequence[x]

            word = self.decode_int(index)
            to_return += str(word)+" "

        return to_return
        # self.tokenizer

    #returns the word whose unique ID is the param
    def decode_int(self, value):
        # print("Decoding "+str(value))
        #searches for word in dictionary
        word = " "
        for key in self.tokenizer.word_index:
            if self.tokenizer.word_index[key]==value:
                word = key
                break

        return word


    def get_token_index(self, token):
        try:
            return self.tokenizer.word_index[token]
        except Exception as error:
            print("Couldn't get ("+str(token)+")'s index: "+str(error))
            return -1


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
    def train(self, X_train, y_train, X_test, y_test):
        # self.model.fit_generator(train_g.__getitem__(), steps_per_epoch= math.ceil(self.trainLen / self.batch_size), epochs=50, 
        #                     validation_data=val_g.__getitem__(),validation_steps=50)

        self.model.fit(X_train, y_train, batch_size = self.batch_size, epochs = 40, shuffle=False)

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



        #makes sure the accuracy calculated previous is the actual accuracy (it is)
        # y_test_pred = self.model.predict(X_test)
        # total_correct = 0
        # for x in range(0, len(y_test_pred)):
        #     index = np.where(y_test_pred[x]==np.amax(y_test_pred[x]))
        #     actual_index = np.where(y_test[x]==1)
        #     if index == actual_index:
        #         total_correct+=1
        # print("Accuracy: "+str(total_correct/len(y_test)))




if __name__ == "__main__":     
    batch_size = 50
    lstm = encoder_decoder_handler(batch_size)
    train_g, val_g, X_test, y_test, embedding_matrix = lstm.create_Emb(100000)
    
    
