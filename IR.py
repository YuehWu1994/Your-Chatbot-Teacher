#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 16:50:31 2019

@author: apple
"""
import numpy as np
from lstmEnc_DNN import lstmEncoder 
from evaluate import countBLEU
from tfidfSentence import tfidfSentence



if __name__ == "__main__": 
    REPEAT_WORD = False    
    batch_size = 50
    lstm = lstmEncoder(batch_size)
    X_train, y_train, X_val, y_val, X_test, y_test, embedding_matrix = lstm.create_Emb(5000)
        
    
    tfidf = tfidfSentence(lstm, X_train, y_train)
    y = tfidf.transformAll(X_train)
    
    
    # evaluate LSTM
    total, correct = 100, 0
        
    bleu = countBLEU(lstm)
    
    for i in range(total):
        tfidf.findMostSimilarOnAll(X_test[i])
        tfidf.findMostSimilarOnOneClass(X_test[i], np.argmax(y_test[i]))
        print('\n')
    
    
    
    