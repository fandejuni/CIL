# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 17:51:26 2019

@author: Justin
"""

import numpy
from helper import PrintProgress, CountFileLines
from nltk.stem.snowball import SnowballStemmer

STEMMER = SnowballStemmer("english")
def GetStem(token):
    return STEMMER.stem(token)

def LoadTokenDict(filename, max_words = 5000):
    token_dict = {}
    
    num_lines = CountFileLines(filename)
    prog_text = "Reading file '{}'".format(filename)
    prog_threshold = num_lines/100
    PrintProgress(prog_text, 0)
    with open(filename, encoding="utf8") as file:
        for i, line in enumerate(file):
            if i>prog_threshold:
                prog_threshold += num_lines/100
                PrintProgress(prog_text, i*100//num_lines)
            token, n = line.split()
            token = GetStem(token)
            n = int(n)
            if token not in token_dict:
                token_dict[token] = 0
            token_dict[token] += n
    PrintProgress(prog_text, -1)
    
    token_list = list(token_dict)
    token_list.sort(key = lambda x:token_dict[x], reverse = True)
    
    if max_words < len(token_list):
        token_list = token_list[:max_words]
    
    return {token:i+1 for (i, token) in enumerate(token_list)}

def LoadTestData(filename, token_dict, line_len = 30):
    
    num_lines = CountFileLines(filename)
    
    X = numpy.zeros((num_lines, line_len), dtype = numpy.int)
        
    prog_text = "Reading file '{}'".format(filename)
    prog_threshold = num_lines/100
    PrintProgress(prog_text, 0)
    with open(filename, encoding="utf8") as file:
        for i, line in enumerate(file):
            x = numpy.zeros(line_len, dtype = numpy.int)
            if i>prog_threshold:
                prog_threshold += num_lines/100
                PrintProgress(prog_text, i*100//num_lines)
            for j, token in enumerate(line.split()):
                if j >= line_len:
                    break
                token = GetStem(token)
                if token in token_dict:
                    x[j] = token_dict[token]
            X[i,:] = x
            
    PrintProgress(prog_text, -1)
    
    return X


def LoadData(filename_pos, filename_neg, token_dict, line_len = 30, prop = 0.8):
    
    num_pos = CountFileLines(filename_pos)
    num_neg = CountFileLines(filename_neg)
    num_tot =  num_pos + num_neg
    X = numpy.zeros((num_tot, line_len), dtype = numpy.int)
    
    xi = 0
    for filename in (filename_pos, filename_neg):
        num_lines = CountFileLines(filename)
        prog_text = "Reading file '{}'".format(filename)
        prog_threshold = num_lines/100
        PrintProgress(prog_text, 0)
        with open(filename, encoding="utf8") as file:
            for i, line in enumerate(file):
                x = numpy.zeros(line_len, dtype = numpy.int)
                if i>prog_threshold:
                    prog_threshold += num_lines/100
                    PrintProgress(prog_text, i*100//num_lines)
                for j, token in enumerate(line.split()):
                    if j >= line_len:
                        break
                    token = GetStem(token)
                    if token in token_dict:
                        x[j] = token_dict[token]
                X[xi,:] = x
                xi += 1
        PrintProgress(prog_text, -1)
    y = numpy.zeros(num_tot, dtype = numpy.int)
    y[0:num_pos] = 1
    
    
    num_training = int(num_tot*prop)
    indices = numpy.random.permutation(num_tot)
    training_idx, test_idx = indices[:num_training], indices[num_training:]
    
    X_train, X_test = X[training_idx], X[test_idx]
    y_train, y_test = y[training_idx], y[test_idx]
    
    
    return (X_train, y_train), (X_test, y_test)


    