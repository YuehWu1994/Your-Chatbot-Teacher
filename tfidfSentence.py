#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 16:59:00 2019

@author: apple
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.gleu_score import sentence_gleu

import numpy as np

class tfidfSentence:
    def __init__(self, lstm, x_train, y_train):
        self.max_train_len = lstm.max_train_len
        self.index_word = lstm.index_word
        self.word_index = lstm.word_index
        self.num_classes = lstm.num_classes
        self.train_text = []
        self.train_text_byClass = [[] for _ in range(self.num_classes)]
        self.genSentence(x_train, y_train)
        
        
    
    
    def genSentence(self, x_train, y_train):
        print("FINDING SIMILAR SENTENCE...")
        for i, encs in enumerate(x_train):
            sentence = "";
            for enc in encs:
                if(enc != 0):
                    sentence = sentence + self.index_word[enc] + " "
            self.train_text.append(sentence)     
            self.train_text_byClass[np.argmax(y_train[i])].append(sentence)
        
 
    # find most similar sentences on all dataset    
    def transformAll(self, x_train):
        tfidf = TfidfVectorizer()
        tfidf_cluster = tfidf.fit_transform(self.train_text)
        tfidf_corpus = tfidf.transform(self.train_text)
        
        cos_similarity = np.dot(tfidf_corpus, tfidf_cluster.T).A
        # exclude itself
        for i in range(len(cos_similarity)):
            cos_similarity[i][i] = 0
            
        c = np.argmax(cos_similarity, axis = 0) 
        
        target = []
        for i in c:
            target.append(x_train[i])
        return  np.array(target)
    
    
    # find most similar sentence on labeled sub-class
    def findMostSimilarOnAll(self, encs):
        sentence = "";
        for enc in encs:
            if(enc != 0):
                sentence = sentence + self.index_word[enc] + " "        
        
        tfidf = TfidfVectorizer()
        tfidf_cluster = tfidf.fit_transform(self.train_text)
        tfidf_corpus = tfidf.transform([sentence])
        
        cos_similarity = np.dot(tfidf_corpus, tfidf_cluster.T).A
        c = np.argmax(cos_similarity)
        
        print("=== Most similar sentence to the input from all subreddit. ===")
        print('Q: ', sentence)
        print('A: ', self.train_text[c])
        return c


    def findMostSimilarOnOneClass(self, encs, label):
        sentence = "";
        for enc in encs:
            if(enc != 0):
                sentence = sentence + self.index_word[enc] + " "        
        
        tfidf = TfidfVectorizer()
        oneClass_tfidf_cluster = tfidf.fit_transform(self.train_text_byClass[label])
        tfidf_corpus = tfidf.transform([sentence])
        cos_similarity = np.dot(tfidf_corpus, oneClass_tfidf_cluster.T).A
        c = np.argmax(cos_similarity)
        
        print("=== Most similar sentence to the input from one subreddit. ===")
        print('Q: ', sentence)
        print('A: ', self.train_text_byClass[label][c])
        return c
        
        