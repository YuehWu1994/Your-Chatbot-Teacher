#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 16:59:00 2019

@author: apple
"""
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

class tfidfSentence:
    def __init__(self, lstm):
        self.tfidf = TfidfVectorizer()
        self.max_train_len = lstm.max_train_len
        self.index_word = lstm.index_word
        self.word_index = lstm.word_index
        self.num_classes = lstm.num_classes
    
    
        
    def transform(self, x_train):
        train_text = []
        for encs in x_train:
            sentence = "";
            for enc in encs:
                if(enc != 0):
                    sentence = sentence + self.index_word[enc] + " "
            train_text.append(sentence)
            
        self.tfidf_cluster = self.tfidf.fit_transform(train_text)
        
        
        tfidf_corpus = self.tfidf.transform(train_text)
        
        cos_similarity = np.dot(tfidf_corpus, self.tfidf_cluster.T).A
        for i in range(len(cos_similarity)):
            cos_similarity[i][i] = 0
            
        c = np.argmax(cos_similarity, axis = 0) 
        
        target = []
        for i in c:
            target.append(x_train[i])
        return  np.array(target)