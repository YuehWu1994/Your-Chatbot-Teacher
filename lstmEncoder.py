# python my_script.py --my-config config.txt  
import os
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
import keras.utils as ku
from sklearn.model_selection import train_test_split
import math
import pickle as pkl

# custom module
from DataGenerator import Generator as gen
from AttentionDecoder import AttentionDecoder
import configargparse


class lstmEncoder:
    def __init__(self, batch_size):
        self.args = self._parse_args()
        self.batch_size = batch_size
        # determine if we could load pickle or tokenize
        cwdFiles = os.listdir(os.getcwd())
        if('enc_doc.pkl' not in cwdFiles or 'label.pkl' not in cwdFiles or 'word_index.pkl' not in cwdFiles):  
            self.load_data()
            
        self.encoded_docs = pkl.load( open('./enc_doc.pkl', "rb" ) )
        self.labels = pkl.load(open('./label.pkl', "rb" ))
        self.word_index = pkl.load(open('./word_index.pkl', "rb" ))
        self.num_classes = len(np.unique(self.labels))
        self.vocab_size = len(self.word_index) + 1

    def _parse_args(self):
        p = configargparse.ArgParser()
        p.add('-c', '--config',required=True, is_config_file=True, help='config file path')
        p.add('--embedding_path', required=True)
        p.add('--data_path', required=True)
        args = p.parse_args()
        return args
    
    def set_limitData(self, X_train, y_train, X_val, y_val, X_test, y_test, limit=None,):
        X_train = X_train[:limit]
        y_train = y_train[:limit]
        X_val = X_val[:limit]
        y_val = y_val[:limit]
        X_test = X_test[:limit]
        y_test = y_test[:limit]  
        return X_train, y_train, X_val, y_val, X_test, y_test

    def load_data(self):
        ### load intput text
        # "/Users/apple/Desktop/q2_course/cs272/finalProject/CS272-NLP-Project/data"
        
        print("LOAD_DATA...")
        corpus = pkl.load( open( self.args.data_path, "rb" ) )
        docs = []
        labels = []  
        
        for c in corpus:
            docs.append(c[0])
            labels.append(c[2])
        labels = np.array(labels)
        del corpus 
        
        print("Tokenize...")     
        t = Tokenizer()
        t.fit_on_texts(self.docs)
        
        ### integer encode the documents
        encoded_docs = t.texts_to_sequences(self.docs)
        
        with open('./enc_doc.pkl','wb') as f:
            pkl.dump(encoded_docs,f)
        with open('./label.pkl','wb') as f:
            pkl.dump(self.labels,f)
        with open('./word_index.pkl','wb') as f:
            pkl.dump(t.word_index,f)  
        
    
    def create_Emb(self):
        ### split in random
        print("Shuffling...")
        X_train, X_test, y_train, y_test = train_test_split(self.encoded_docs, self.labels, test_size=0.1, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        
        ### DEBUG: set data length
        X_train, y_train, X_val, y_val, X_test, y_test = self.set_limitData(X_train, y_train, X_val, y_val, X_test, y_test, 1000)
        self.trainLen = len(X_train)
        
        
        ### pad test data
        max_len = max(len(x) for x in X_test)
        X_test = pad_sequences(X_test, maxlen=max_len, padding='post')
        y_test = ku.to_categorical(y_test, num_classes=self.num_classes)


        ### load the whole embedding into memory
        embeddings_index = dict()
        f = open(self.args.embedding_path, encoding="utf-8")
        # "/Users/apple/Desktop/q2_course/cs272/finalProject/glove.6B/glove.6B.100d.txt"
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
                
        train_g = gen(X_train, y_train, self.batch_size, self.num_classes)
        val_g = gen(X_val, y_val, self.batch_size, self.num_classes)
        
        return train_g, val_g, X_test, y_test, embedding_matrix


    def buildModel(self, embedding_matrix):
        self.model = Sequential()
        e = Embedding(self.vocab_size, 100, weights=[embedding_matrix], input_length=None, trainable=False)
        self.model.add(e)
        self.model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
        self.model.add(MaxPooling1D(pool_size=2))
        self.model.add(LSTM(200))
        self.model.add(AttentionDecoder(200, output_dim=self.num_classes))
        self.model.add(Dense(self.num_classes, activation='sigmoid'))
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
        print(self.model.summary())
        return self.model
        
        
    def train(self,  train_g, val_g, X_test, y_test):
        self.model.fit_generator(train_g.__getitem__(), steps_per_epoch= math.ceil(self.trainLen / self.batch_size), epochs=50, 
                            validation_data=val_g.__getitem__(),validation_steps=50)
        
        ### evaluate the model
        #loss, accuracy = model.evaluate(padded_docs, labels, verbose=0)
        loss, accuracy = self.model.evaluate(X_test, y_test)
        
        print('Accuracy: %f' % (accuracy*100))

if __name__ == "__main__":     
    lstm = lstmEncoder(50)
    train_g, val_g, X_test, y_test, embedding_matrix = lstm.create_Emb()
    lstm.buildModel(embedding_matrix)
    lstm.train(train_g, val_g, X_test, y_test)
    
    
