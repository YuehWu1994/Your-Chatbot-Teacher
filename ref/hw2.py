#!/usr/bin/env python
# coding: utf-8

# In[5]:


#!/bin/python

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import sys
import pickle as pkl
import nltk
import itertools

# Python 3 backwards compatibility tricks
if sys.version_info.major > 2:

    def xrange(*args, **kwargs):
        return iter(range(*args, **kwargs))

    def unicode(*args, **kwargs):
        return str(*args, **kwargs)


# In[7]:


def read_texts(tarfname, dname):
    """Read the data from the homework data file.

    Given the location of the data archive file and the name of the
    dataset (one of brown, reuters, or gutenberg), this returns a
    data object containing train, test, and dev data. Each is a list
    of sentences, where each sentence is a sequence of tokens.
    """
    import tarfile
    tar = tarfile.open(tarfname, "r:gz", errors = 'replace')
    train_mem = tar.getmember(dname + ".train.txt")
    train_txt = unicode(tar.extractfile(train_mem).read(), errors='replace').lower()
    test_mem = tar.getmember(dname + ".test.txt")
    test_txt = unicode(tar.extractfile(test_mem).read(), errors='replace').lower()
    dev_mem = tar.getmember(dname + ".dev.txt")
    dev_txt = unicode(tar.extractfile(dev_mem).read(), errors='replace').lower()

    from sklearn.feature_extraction.text import CountVectorizer
    count_vect = CountVectorizer()
    count_vect.fit(train_txt.split("\n"))
    tokenizer = count_vect.build_tokenizer()
    class Data: pass
    data = Data()
    data.train = []
    for s in train_txt.split("\n"):
        toks = tokenizer(s)
        if len(toks) > 0:
            data.train.append(toks)
    data.test = []
    for s in test_txt.split("\n"):
        toks = tokenizer(s)
        if len(toks) > 0:
            data.test.append(toks)
    data.dev = []
    for s in dev_txt.split("\n"):
        toks = tokenizer(s)
        if len(toks) > 0:
            data.dev.append(toks)
            
    word_freq = nltk.FreqDist(itertools.chain(*data.train))
    print("Found %d unique words tokens." % len(word_freq.items()))
    vocab = word_freq.most_common(int(len(word_freq.items())*0.99))
    index_to_word = [x[0] for x in vocab]
    word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])
    
    for i, sent in enumerate(data.train):
        data.train[i] = "<bos> "+" ".join([w if w in word_to_index else "<unk>" for w in sent])+" <eos>"
    for i, sent in enumerate(data.dev):
        data.dev[i] = "<bos> "+" ".join([w if w in word_to_index else "<unk>" for w in sent])+" <eos>"
    for i, sent in enumerate(data.test):
        data.test[i] = "<bos> "+" ".join([w if w in word_to_index else "<unk>" for w in sent])+" <eos>"
    
    print(dname," read.", "train:", len(data.train), "dev:", len(data.dev), "test:", len(data.test))
    return data


# In[8]:


dnames = ["brown", "reuters", "gutenberg"]
tar_name = "data/corpora.tar.gz"
datas = dict()
# Learn the models for each of the domains, and evaluate it
for dname in dnames:
    print("-----------------------")
    print(dname)
    data = read_texts(tar_name, dname)
    tmp = {}
    tmp['train'] = data.train 
    tmp['dev'] = data.dev 
    tmp['test'] = data.test
    datas[dname] = tmp


# In[9]:


with open("clean_full.pkl",'wb') as f:
    pkl.dump(datas,f)


# In[4]:


from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense, Dropout, TimeDistributed, Activation
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential
import keras.utils as ku
import pickle as pkl

from tensorflow import set_random_seed
from numpy.random import seed
set_random_seed(2019)
seed(2020)

import numpy as np
import string, os



# In[5]:


def load_data(dname='brown'):
    with open("clean.pkl",'rb') as f:
        corpus = pkl.load(f)[dname]
    train_text = corpus['train']
    dev_text = corpus['dev']
    test_text = corpus['test']
    tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;=?@[\]^_`{|}~ ')
    tokenizer.fit_on_texts(train_text)
    train_data = tokenizer.texts_to_sequences(train_text)
    dev_data = tokenizer.texts_to_sequences(dev_text)
    test_data = tokenizer.texts_to_sequences(test_text)
    vocab_size = len(tokenizer.word_index)+1
    reversed_dictionary = dict(zip(tokenizer.word_index.values(), tokenizer.word_index.keys()))
    return train_data, dev_data, test_data, vocab_size, reversed_dictionary


# In[6]:


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


# In[7]:


train_data, valid_data, test_data, vocab_size, reversed_dictionary = load_data()

batch_size = 10
hidden_size = 100
num_epochs = 10
# train_data = train_data[:100]
# valid_data = valid_data[:100]
train_data_generator = Generator(train_data, batch_size, vocab_size)
valid_data_generator = Generator(valid_data, len(valid_data), vocab_size)
checkpointer = ModelCheckpoint(filepath='./model/' + '/model-{epoch:02d}.hdf5', verbose=1)


# In[ ]:


model = Sequential()
model.add(Embedding(vocab_size,hidden_size,input_length = None))
model.add(LSTM(hidden_size, return_sequences=True))
model.add(LSTM(hidden_size, return_sequences=True))
model.add(Dropout(0.2))
model.add(TimeDistributed(Dense(vocab_size)))
# model.add(Activation('softmax'))
print(model.summary())
model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=['categorical_accuracy'] )
model.fit_generator(train_data_generator.__getitem__(), steps_per_epoch = len(train_data)//(batch_size), epochs=num_epochs
,validation_data=valid_data_generator.__getitem__(),validation_steps=len(valid_data),shuffle=True, callbacks=[checkpointer])


# In[40]:


def read_texts_for_testing(tarfname, dname, vocab_size):
    """Read the data from the homework data file.

    Given the location of the data archive file and the name of the
    dataset (one of brown, reuters, or gutenberg), this returns a
    data object containing train, test, and dev data. Each is a list
    of sentences, where each sentence is a sequence of tokens.
    """
    import tarfile
    tar = tarfile.open(tarfname, "r:gz", errors = 'replace')
    train_mem = tar.getmember(dname + ".train.txt")
    train_txt = unicode(tar.extractfile(train_mem).read(), errors='replace').lower()
    test_mem = tar.getmember(dname + ".test.txt")
    test_txt = unicode(tar.extractfile(test_mem).read(), errors='replace').lower()
    
    dnames=["brown","reuters","gutenberg"]
    out = [name for name in dnames if name != dname]
    
    out1_mem = tar.getmember(out[0]+".test.txt")
    out1_text = unicode(tar.extractfile(out1_mem).read(), errors='replace').lower()
    
    out2_mem = tar.getmember(out[1]+".test.txt")
    out2_text = unicode(tar.extractfile(out2_mem).read(), errors='replace').lower()
    
    from sklearn.feature_extraction.text import CountVectorizer
    count_vect = CountVectorizer()
    count_vect.fit(train_txt.split("\n"))
    tokenizer = count_vect.build_tokenizer()
    class Data: pass
    data = Data()
    data.train = []
    for s in train_txt.split("\n"):
        toks = tokenizer(s)
        if len(toks) > 0:
            data.train.append(toks)
    data.test = []
    for s in test_txt.split("\n"):
        toks = tokenizer(s)
        if len(toks) > 0:
            data.test.append(toks)
    out1 = []
    for s in out1_text.split("\n"):
        toks = tokenizer(s)
        if len(toks)>0:
            out1.append(toks)
    out2 = []
    for s in out2_text.split("\n"):
        toks = tokenizer(s)
        if len(toks)>0:
            out2.append(toks)
    
    word_freq = nltk.FreqDist(itertools.chain(*data.train))
    print("Found %d unique words tokens." % len(word_freq.items()))
    vocab = word_freq.most_common(vocab_size)
    print("in {} vocab size=".format(dname),len(vocab))
    index_to_word = [x[0] for x in vocab]
    word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])
    
    for i, sent in enumerate(data.train):
        data.train[i] = "<bos> "+" ".join([w if w in word_to_index else "<unk>" for w in sent])+" <eos>"
    for i, sent in enumerate(data.test):
        data.test[i] = "<bos> "+" ".join([w if w in word_to_index else "<unk>" for w in sent])+" <eos>"
    for i,sent in enumerate(out1):
        out1[i] = "<bos> "+" ".join([w if w in word_to_index else "<unk>" for w in sent])+" <eos>"
    for i,sent in enumerate(out2):
        out2[i] = "<bos> "+" ".join([w if w in word_to_index else "<unk>" for w in sent])+" <eos>"
    
    tests = {}
    tests["self"] = data.train
    tests["in"] = data.test
    tests["out1"] = out1
    tests["out2"] = out2
    with open("test_for_"+dname+".pkl",'wb') as f:
        pkl.dump(tests,f)
    


# In[89]:


# 36745, 26763, 36098
dnames = ["brown", "reuters", "gutenberg"]
vocab_size = [36743,26758,36298]
tarfname = "data/corpora.tar.gz"
for i,d in enumerate(dnames):
    read_texts_for_testing(tarfname, d,vocab_size[i])


# In[90]:


with open("test_for_gutenberg.pkl","rb") as f:
    test = pkl.load(f)
    


# In[77]:


len(test['self'])


# In[91]:


from keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;=?@[\]^_`{|}~ ')
tokenizer.fit_on_texts(test['self'])
print(len(tokenizer.word_index))
max([max(i) for i in tokenizer.texts_to_sequences(test['self'])])


# In[ ]:




