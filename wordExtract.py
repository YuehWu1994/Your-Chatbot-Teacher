#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 15:15:47 2019

@author: Yueh
"""
import numpy as np
import pandas as pd


class WordExtract:
    def __init__ (self, path):
        # path
        self.path = path
        
        # column index
        self.TEXT = 2
        self.META_REDDIT = 4
        self.SUB_REDDIT = 5
        self.UP = 8
        self.DOWN = 9
        self.LINK_KARMA = 10
        self.KARMA = 11
        self.IS_GOLD = 12
        
        # extracted info
        self.text = []
        self.meta = ""
        self.sub = ""
        self.up = []
        self.down = []
        self.link_karma = []
        self.karma = []
        self.is_gold = []
        
    def extract(self):
        df = pd.read_csv(self.path, encoding="big5")
        rawData = df.as_matrix()
        row = len(df)
        
        self.meta = rawData[1][self.META_REDDIT]
        self.sub = rawData[1][self.SUB_REDDIT]

        for i in range (1, row):
            self.text.append(rawData[i][self.TEXT])
            self.up.append(rawData[i][self.UP])
            self.down.append(rawData[i][self.DOWN])
            self.link_karma.append(rawData[i][self.LINK_KARMA])
            self.karma.append(rawData[i][self.KARMA])
            self.is_gold.append(rawData[i][self.IS_GOLD])    
            
#### example usage            
#W_E = WordExtract("../reddit-dataset/entertainment_anime.csv")
#W_E.extract()