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
import pickle

from DataGenerator import Generator as gen


corpus = pickle.load( open( "/Users/apple/Desktop/q2_course/cs272/finalProject/CS272-NLP-Project/data", "rb" ) )

docs = []
labels = []
batch_size = 50

for c in corpus:
    docs.append(c[0])
    labels.append(c[2])

labels = np.array(labels)
num_classes = len(np.unique(labels))
del corpus

docs = docs[:250]
labels = labels[:250]



### prepare tokenizer
t = Tokenizer()
t.fit_on_texts(docs)
vocab_size = len(t.word_index) + 1

### integer encode the documents
encoded_docs = t.texts_to_sequences(docs)
#print(encoded_docs)


### split in random
X_train, X_val, y_train, y_val = train_test_split(encoded_docs[:150], labels[:150], test_size=0.33, random_state=42)


### pad documents to a max length of 4 words
#max_length = 4
X_test = pad_sequences(encoded_docs[150:], maxlen=100, padding='post')
y_test = ku.to_categorical(labels[150:], num_classes=num_classes)



### load the whole embedding into memory
embeddings_index = dict()
f = open('/Users/apple/Desktop/q2_course/cs272/finalProject/glove.6B/glove.6B.100d.txt', encoding="utf-8")
for line in f:
	values = line.split()
	word = values[0]
	coefs = np.asarray(values[1:], dtype='float32')
	embeddings_index[word] = coefs
f.close()
print('Loaded %s word vectors.' % len(embeddings_index))

### create a weight matrix for words in training docs
embedding_matrix = np.zeros((vocab_size, 100))
for word, i in t.word_index.items():
	embedding_vector = embeddings_index.get(word)
	if embedding_vector is not None:
		embedding_matrix[i] = embedding_vector
        
train_g = gen(X_train, y_train, batch_size, num_classes)
val_g = gen(X_val, y_val, batch_size, num_classes)


### define model
model = Sequential()
### input length is dynamic
e = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=100, trainable=False)
model.add(e)
#model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
#model.add(MaxPooling1D(pool_size=2))
#model.add(LSTM(200))
model.add(Flatten())
model.add(Dense(num_classes, activation='sigmoid'))

### compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

### summarize the model
print(model.summary())

### fit the model
#model.fit(padded_docs, labels, epochs=50, verbose=0)
model.fit_generator(train_g.__getitem__(), steps_per_epoch= math.ceil(len(docs) / batch_size), epochs=50, 
                    validation_data=val_g.__getitem__(),validation_steps=len(X_val), verbose=0)

### evaluate the model
#loss, accuracy = model.evaluate(padded_docs, labels, verbose=0)
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

print('Accuracy: %f' % (accuracy*100))


