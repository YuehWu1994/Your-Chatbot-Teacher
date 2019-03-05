# LSTM and CNN for sequence classification in the IMDB dataset
import numpy
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer

# fix random seed for reproducibility
numpy.random.seed(7)
# load the dataset but only keep the top n words, zero the rest

docs = ['postings about blah blah','nice post']
labels = [0,0]

t = Tokenizer()
t.fit_on_texts(docs)
vocab_size = len(t.word_index)+1

encoded_docs  = t.texts_to_sequences(docs)
# truncate and pad input sequences
max_comment_length = 100
X_train = sequence.pad_sequences(X_train, maxlen=max_comment_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_comment_length)
# this may be useful, idk lol

from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense, Dropout, TimeDistributed, Activation
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential
import keras.utils as ku
import pickle as pkl
from tensorflow import set_random_seed
import numpy as np
from numpy.random import seed
import string, os

class Generator(object):

    def __init__(self,data,batch_size,vocab_size):
        self.data = data
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.inds = np.arange(len(data))
        self.iter = 0
    def __len__(self):
        return int(np.floor(len(self.data)/self.batch_size))
    def __getitem__(self):
        while True:
            input_seq = []
            id_list = self.inds[self.iter*self.batch_size:(self.iter+1)*self.batch_size]
            for i in range(self.batch_size):
                input_seq.append(self.data[id_list[i]])
            max_len = max(len(x) for x in input_seq)
            padded = pad_sequences(input_seq, maxlen=max_len, padding='pre')
            X = np.array(padded)[:,:-1]
            y = np.zeros((self.batch_size,max_len-1,self.vocab_size))
            for i in range(self.batch_size):
                y[i,:,:] = ku.to_categorical(padded[i][1:], num_classes=self.vocab_size)
            self.iter+=1
            if self.iter>=len(self.data)//self.batch_size:
                self.iter=0
                np.random.shuffle(self.inds)
            yield X,y



with open('../glove_data/glove.6B/glove.6B.100d.txt') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
print('Loaded %s word vectors.' % len(embeddings_index))
embedding_matrix = np.zeros((vocab_size, 100))
for word, i in t.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

# create the model
embedding_vecor_length = 100
model = Sequential()
model.add(Embedding(vocab_size, embedding_vecor_length, input_length=max_comment_length))
model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(LSTM(200))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, epochs=3, batch_size=64)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))