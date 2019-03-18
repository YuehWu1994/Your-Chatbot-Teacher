#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.bleu_score import sentence_bleu



class countBLEU:
    def __init__(self, lstm):
        self.comment = 0
        self.sentenceScore = 0
        self.corpustenceScore = 0
        self.max_train_len = lstm.max_train_len
        self.index_word = lstm.index_word
        
    '''
    def genTestCorpus(y_t):
    def count_corpus_BLEU():   
    '''
    def count_sentence_BLEU(self, y, target):
        reference = [[]]
        candidate = []
        for i in range(self.max_train_len):
            if(target[i] != 0):
                candidate.append(self.index_word[target[i]])
            if(y[i] != 0):
                reference[0].append(self.index_word[y[i]])
        score = sentence_bleu(reference, candidate)
        self.sentenceScore += score
        self.comment += 1
        print(score)
        
