#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.gleu_score import sentence_gleu



# class countBLEU:
#     def __init__(self, lstm):
#         self.comment = 0
#         self.bleuScore = 0.0
#         self.gleuScore = 0.0
#         #self.classtenceScore = 0
#         self.max_train_len = lstm.max_train_len
#         self.index_word = lstm.index_wor


#     def count_BLEU(self, y, target):
#         reference = [[]]
#         candidate = []
#         for i in range(self.max_train_len):
#             if(target[i] != 0):
#                 candidate.append(self.index_word[target[i]])
#             if(y[i] != 0):
#                 reference[0].append(self.index_word[y[i]])

#         score = sentence_bleu(reference, candidate)
#         g_score = sentence_gleu(reference, candidate)
        
#         self.bleuScore += score
#         self.gleuScore += g_score

#         self.comment += 1
#         print(score)
#         print(g_score)

        
class countBLEU:
    def __init__(self):
        self.comment = 0
        self.bleuScore = 0.0
        self.gleuScore = 0.0
        #self.classtenceScore = 0


    def count_BLEU(self, y, target):
        reference = [[]]
        candidate = []
        for i in range(self.max_train_len):
            if(target[i] != 0):
                candidate.append(target[i])
            if(y[i] != 0):
                reference[0].append(y[i])

        score = sentence_bleu(reference, candidate)
        g_score = sentence_gleu(reference, candidate)
        
        self.bleuScore += score
        self.gleuScore += g_score

        self.comment += 1
        print("BLEU score: "+str(score))
        print("GLEU score: "+str(g_score))


