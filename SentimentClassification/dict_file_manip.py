# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 17:57:37 2019

@author: Justin
"""

from helper import PrintProgress, CountFileLines

def CreateDictFromFile(filename, output_filename, threshold):
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
            for token in line.split():
                if token not in token_dict:
                    token_dict[token] = 0
                token_dict[token] += 1
    PrintProgress(prog_text, -1)
    
    num_lines = len(token_dict)
    prog_text = "Writing to file '{}'".format(output_filename)
    prog_threshold = num_lines/100
    PrintProgress(prog_text, 0)            
    with open(output_filename, "w", encoding="utf8") as file:    
        for i, token in enumerate(token_dict):
            if i>prog_threshold:
                prog_threshold += num_lines/100
                PrintProgress(prog_text, i*100//num_lines)
            c = token_dict[token]
            if c > threshold:
                file.write("{} {}\n".format(token, c))
    PrintProgress(prog_text, -1)

def AddDictFiles(filenames, output_filename):    
    token_dict = {}
    
    for filename in filenames:
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
                n = int(n)
                if token not in token_dict:
                    token_dict[token] = 0
                token_dict[token] += n
        PrintProgress(prog_text, -1)
    
    token_list = list(token_dict)
    token_list.sort(key = lambda x:token_dict[x], reverse = True)
        
    num_lines = len(token_list)
    prog_text = "Writing to file '{}'".format(output_filename)
    prog_threshold = num_lines/100   
    PrintProgress(prog_text, 0)          
    with open(output_filename, "w", encoding="utf8") as file:    
        for i, token in enumerate(token_list):
            if i>prog_threshold:
                prog_threshold += num_lines/100
                PrintProgress(prog_text, i*100//num_lines)
            file.write("{} {}\n".format(token, token_dict[token]))
    PrintProgress(prog_text, -1)