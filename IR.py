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
from keras.models import load_model

if __name__ == "__main__": 
    REPEAT_WORD = False    
    batch_size = 50
    lstm = lstmEncoder(batch_size)
    X_train, y_train, X_val, y_val, X_test, y_test, embedding_matrix = lstm.create_Emb(300000)
        
    
    tfidf = tfidfSentence(lstm, X_train, y_train)
    
    
    # evaluate LSTM
    total, correct = 100, 0
        
    bleu = countBLEU(lstm)
    classifier = load_model('./classifier.h5')
    loss, accuracy = classifier.evaluate(X_test, y_test)
    print("loss %f " % (loss*100))
    print('Accuracy: %f' % (accuracy*100))
    pred = classifier.predict(X_test)
    
    
    '''
    # input a comment here
    c = "hey how are you"
    enc = np.zeros(lstm.max_train_len)
    
    c_split = c.split()
    for i in range(0, min(20, len(c_split))):
        enc[lstm.max_train_len-1-i] = lstm.word_index[c_split[len(c_split)-1-i]]
    classifier.predict(np.reshape(enc, (1, lstm.max_train_len)))
    '''    
    
        
    with open('IR.csv', mode='w') as f:
        ir_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        ir_writer.writerow(['#id', 'question', 'most similar comment from all the subreddit', 'most similar comment from 1 the subreddit'])
    
        for i in range(total):
            c, q, a = tfidf.findMostSimilarOnAll(X_test[i])
            c1, q1, a1 = tfidf.findMostSimilarOnOneClass(X_test[i], np.argmax(pred[i]))
            print('\n')
    
            ir_writer.writerow([str(i), q, a, a1])
        f.close();