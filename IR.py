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
    '''
    loss, accuracy = classifier.evaluate(X_test, y_test)
    print("loss %f " % (loss*100))
    print('Accuracy: %f' % (accuracy*100))
    '''
    pred = classifier.predict(X_test)
    
    
    '''
    d = pkl.load( open('./label_dict.pkl', "rb" ) )
    
    sentence = ""
    for num in X_test[0]:
        if num != 0:
            sentence = sentence + lstm.index_word[num] + ' '
    # input a comment here
    c = sentence
    enc = np.zeros(lstm.max_train_len)
    
    c_split = c.split()
    for i in range(0, min(20, len(c_split))):
        enc[lstm.max_train_len-1-i] = lstm.word_index[c_split[len(c_split)-1-i]]
    prob = classifier.predict(np.reshape(enc, (1, lstm.max_train_len)))
    p = np.argmax(prob)
    print(d[p], prob[0][p])
    '''    
    
    # create csv header
    header = ['#id', 'question', 'q_subclass']
    for i in range(10):
        header.append(str(i))
        header.append("subclass"+str(i))
    header.append('most similar comment from 1 the subreddit')
    
    with open('IR.csv', mode='w') as f:
        ir_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        ir_writer.writerow(header)
    
        for i in range(total):
            c, q, output = tfidf.find10MostSimilarOnAll(X_test[i])
            if(len(output) != 20):
                continue
            content = [str(i), q, str(np.argmax(y_test[i]))]
            for o in output:
                content.append(o)   
            c1, q1, a1 = tfidf.findMostSimilarOnOneClass(X_test[i], np.argmax(pred[i]))
            print('\n')
            content.append(a1)
            
            ir_writer.writerow(content)
        f.close();