#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.bleu_score import sentence_bleu



class countBLEU:
    def __init__(self, lstm):
        self.comment = 0
        self.sentenceScore = 0
        self.classtenceScore = 0
        self.max_train_len = lstm.max_train_len
        self.index_word = lstm.index_word
        self.num_classes = lstm.num_classes
        #from itertools import repeat
        #self.corpus = [[] for i in repeat(None, self.num_classes)]
        
    '''
    def genTestCorpus(self, testEnc, test_label):
        for i, enc in enumerate(testEnc):
            label = np.argmax(test_label[i])
            comment = []
            for j in range(self.max_train_len):
                if(enc[i][j] != 0):
                    comment.append(self.index_word[enc[i][j]])
            self.corpus[label].append(comment)
    '''       
    def count_BLEU(self, y, target, label):
        reference = [[]]
        candidate = []
        for i in range(self.max_train_len):
            if(target[i] != 0):
                candidate.append(self.index_word[target[i]])
            if(y[i] != 0):
                reference[0].append(self.index_word[y[i]])

        score = sentence_bleu(reference, candidate)
        #classScore = corpus_bleu(self.corpus[label], candidate)
        
        self.sentenceScore += score
        #self.classtenceScore += classScore
        self.comment += 1
        print(len(self.corpus[label]))
        print(score)
        
