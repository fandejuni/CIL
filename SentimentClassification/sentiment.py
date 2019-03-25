# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 16:41:43 2019

@author: Justin
"""


class Token:
    def __init__(self, string, sentiment = 0, back_enhancement = 0, 
                 front_enhancement = 0, transmission = 0.5):
        self.string = string
        self.sentiment = sentiment
        
        self.back_enhancement = back_enhancement
        self.front_enhancement = front_enhancement
        self.transmission = transmission

def revenumerate(seq):
    return ((len(seq)-1-i, x) for i, x in enumerate(reversed(seq)))

def classify(sentence):
    enhancement = [0 for _ in sentence]
    for i, tok in enumerate(sentence):
        if(i+1 == len(sentence)):
            break
        enhancement[i+1] += tok.back_enhancement*(1+enhancement[i])
        print(tok.string, tok.back_enhancement, sentence[i+1].string, sentence[i+1].sentiment)
        enhancement[i+1] += enhancement[i]*tok.transmission
    tot = sum(tok.sentiment*(0.5+enhancement[i]) for i,tok in enumerate(sentence))
    
    enhancement = [0 for _ in sentence]
    for i, tok in revenumerate(sentence):
        if(i-1 == 0):
            break
        enhancement[i-1] += tok.front_enhancement*(1+enhancement[i])
        enhancement[i-1] += enhancement[i]*tok.transmission  
    tot += sum(tok.sentiment*(0.5+enhancement[i]) for i,tok in enumerate(sentence))
    
    return tot

    
        
    
_this = Token("this", 0, 0, 0, 0.8)
_is = Token("is", 0, 0, 0, 0.1)
_not = Token("not", 0, -2, 0, 0.8)
_the = Token("the", 0, 0, 0, 0.8)
_very = Token("very", 0, 2, 0, 0.8)
_best = Token("best", 1, 0, 0, 0.5)
_worst = Token("worst", -1, 0, 0, 0.5)

s = [_this, _is, _very, _the, _the, _the, _the, _best]

classify(s)