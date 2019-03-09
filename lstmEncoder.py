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
from sklearn.model_selection import train_test_split
import math
import pickle
from DataGenerator import Generator as gen
# from keras_wc_embd import get_dicts_generator, WordCharEmbd, get_embedding_weights_from_file
from wrapper import WordCharEmbd
import configargparse
import keras
from keras.callbacks import ModelCheckpoint
from random import shuffle
class lstmEncoder:
    def __init__(self, batch_size):
        self.args = self._parse_args()
        self.docs, self.labels = self.load_data()
        self.batch_size = batch_size
        self.num_classes = len(np.unique(self.labels))
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

    def _exp(self,size_limit=None):
        shuf_ind = [i for i in range(len(self.labels))]
        shuffle(shuf_ind)
        self.docs = [self.docs[i] for i in shuf_ind]
        self.labels = [self.labels[i] for i in shuf_ind]
        del shuf_ind
        self.docs = self.docs[:size_limit]
        self.labels = self.labels[:size_limit]
        # self.docs = [d.split(' ')[:20] for d in self.docs]
        wc_emb = WordCharEmbd( 
        word_min_freq=2,
        char_min_freq=2,
        word_ignore_case=False,
        char_ignore_case=False,)
        for doc in self.docs:
            wc_emb.update_dicts(doc.split(' '))
        # word_embd_weights = get_embedding_weights_from_file(wc_emb.get_dicts()[0], self.args.embedding_path, ignore_case=True)
        # print(word_embd_weights.shape)
        inputs, embd_layer = wc_emb.get_embedding_layer(
        word_embd_dim=100,
        char_embd_dim=30,
        char_hidden_dim=150,
        word_embd_file_path=self.args.embedding_path,
        word_embd_trainable=True,
        char_hidden_layer_type='lstm')
       
        lstm_layer = LSTM(units=50)(embd_layer)
        prelu_activated = PReLU()(lstm_layer)
        dropout1 = Dropout(0.1)(prelu_activated)
        dense1 = Dense(50,activation='selu')(dropout1)
        dropout2 = Dropout(0.1)(dense1)
        dense2 = Dense(50,activation='selu')(dropout2)
        dropout3 = Dropout(0.1)(dense2)
        softmax_layer = Dense(self.num_classes, activation='softmax')(dropout3)
        model = keras.models.Model(inputs=inputs, outputs=softmax_layer)
        model.compile(
        optimizer='adam',
        loss=keras.losses.categorical_crossentropy,
        metrics=[keras.metrics.categorical_accuracy],)
        model.summary()
        self.docs = [d.split(' ')[:20] for d in self.docs]
        print(self.docs[0])
        
        X_train, X_test, y_train, y_test = train_test_split(self.docs, self.labels, test_size=0.1, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        
        
        y_train = ku.to_categorical(y_train, num_classes=self.num_classes)
        y_test = ku.to_categorical(y_test, num_classes=self.num_classes)
        y_val = ku.to_categorical(y_val, num_classes=self.num_classes)
        def train_batch_generator():
            start = 0
            end = self.batch_size
            while True:
                yield wc_emb.get_batch_input(X_train[start:end]), np.array(y_train[start:end])
                if end+self.batch_size>len(X_train):
                    start = 0
                    end = self.batch_size
                else:
                    start,end = start+self.batch_size,end+self.batch_size
        
        def dev_batch_generator():
            start = 0
            end = 200
            while True:
                yield wc_emb.get_batch_input(X_val[start:end]), np.array(y_val[start:end])
                if end+200>len(X_test):
                    start = 0
                    end = 200
                else:
                    start,end = start+200,end+200
       

        filepath = "wc_emb_best.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        callbacks=[checkpoint]
        model.fit_generator(generator=train_batch_generator(), steps_per_epoch= math.ceil(len(self.docs) / self.batch_size), epochs=5, 
                            validation_data=dev_batch_generator(),validation_steps=50, callbacks=callbacks)

        loss, accuracy = model.evaluate(wc_emb.get_batch_input(X_test), y_test)
        print("loss %f " % (loss*100))
        print('Accuracy: %f' % (accuracy*100))
        y_hat = model.predict(wc_emb.get_batch_input(X_test))
        import pdb;pdb.set_trace()
        print('test_acc:',sum(np.where( np.argmax(y_hat) == y_test) )/len(y_test) )
        return


    def load_data(self):
        ### load intput text
        # "/Users/apple/Desktop/q2_course/cs272/finalProject/glove.6B/glove.6B.100d.txt"
        
        print("LOAD_DATA...")
        corpus = pickle.load( open( self.args.data_path, "rb" ) )
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
        
        print("X_train: "+str(len(X_train)))
        print("y_train: "+str(len(y_train)))
        print("X_val: "+str(len(X_val)))
        print("y_val: "+str(len(y_val)))
        print("X_test: "+str(len(X_test)))
        print("y_test: "+str(len(y_test)))


        ### DEBUG: set data length
        X_train, y_train, X_val, y_val, X_test, y_test = self.set_limitData( X_train, y_train, X_val, y_val, X_test, y_test, 200000)
        self.trainLen = len(X_train)

        

        ### pad train data
        # self.max_train_len = max(len(x) for x in X_train)
        self.max_train_len = 20
        print("max_train_len: "+str(self.max_train_len))
        X_train = pad_sequences(X_train, maxlen=self.max_train_len, padding='pre')
        y_train = ku.to_categorical(y_train, num_classes=self.num_classes)
        
        
        ### pad test data
        # self.max_test_len = max(len(x) for x in X_test)
        # self.max_test_len = 1000
        self.max_test_len = self.max_train_len
        print("max_test_len: "+str(self.max_test_len))
        X_test = pad_sequences(X_test, maxlen=self.max_test_len, padding='pre')
        y_test = ku.to_categorical(y_test, num_classes=self.num_classes)


        ### load the whole embedding into memory
        embeddings_index = dict()
        f = open(self.args.embedding_path, encoding="utf-8")
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
        return X_train, y_train, X_test, y_test, embedding_matrix


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

        self.model.fit(X_train, y_train, batch_size = self.batch_size, epochs = 10, shuffle=False)

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
    lstm = lstmEncoder(batch_size)
    # train_g, val_g, X_test, y_test, embedding_matrix = lstm.create_Emb()
    # lstm.buildModel(embedding_matrix)
    # lstm.train(train_g, val_g, X_test, y_test)
    lstm._exp(2000)
    
    
