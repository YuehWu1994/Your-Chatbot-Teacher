#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 15:15:47 2019


@author: Yueh
"""
import numpy as np
import pandas as pd
import math

class WordExtract:
    def __init__ (self, path):
        # path
        df = pd.read_csv(path, encoding="big5")
        self.rawData = df.as_matrix()
        self.column = int(self.rawData[0].size)
        self.row = len(df)
        
        
        # column index
        self.TEXT = self.column - 11
        self.META_REDDIT = self.column - 9
        self.SUB_REDDIT = self.column - 8
        self.UP = self.column - 5
        self.DOWN = self.column - 4
        self.LINK_KARMA = self.column - 3
        self.KARMA = self.column - 2
        self.IS_GOLD = self.column - 1
        
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
        self.meta = self.rawData[1][self.META_REDDIT]
        self.sub = self.rawData[1][self.SUB_REDDIT]

        for i in range (1, self.row):
            if not isinstance(self.rawData[i][self.TEXT], str):
                #print("NAN")
                continue
        
            self.text.append(self.rawData[i][self.TEXT])
            self.up.append(self.rawData[i][self.UP])
            self.down.append(self.rawData[i][self.DOWN])
            self.link_karma.append(self.rawData[i][self.LINK_KARMA])
            self.karma.append(self.rawData[i][self.KARMA])
            self.is_gold.append(self.rawData[i][self.IS_GOLD])    
            
#### example usage    
if __name__ == "__main__":        
    W_E = WordExtract("../reddit-dataset/humor_funny.csv")
    W_E.extract()
