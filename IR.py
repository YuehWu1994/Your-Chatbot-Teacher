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
import csv



if __name__ == "__main__": 
    REPEAT_WORD = False    
    batch_size = 50
    lstm = lstmEncoder(batch_size)
    X_train, y_train, X_val, y_val, X_test, y_test, embedding_matrix = lstm.create_Emb(300000)
        
    
    tfidf = tfidfSentence(lstm, X_train, y_train)
    
    
    # evaluate LSTM
    total, correct = 100, 0
        
    bleu = countBLEU(lstm)
    
    with open('IR.csv', mode='w') as f:
        ir_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        ir_writer.writerow(['#id', 'question', 'most similar comment from all the subreddit', 'most similar comment from 1 the subreddit'])
    
        for i in range(total):
            c, q, a = tfidf.findMostSimilarOnAll(X_test[i])
            c1, q1, a1 = tfidf.findMostSimilarOnOneClass(X_test[i], np.argmax(y_test[i]))
            print('\n')
    
            ir_writer.writerow([str(i), q, a, a1])
        f.close();