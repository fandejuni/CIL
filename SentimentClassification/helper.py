# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 17:54:57 2019

@author: Justin
"""

def PrintProgress(text, prog):
    if prog == -1:
        print("\r{} [Done]".format(text, prog),flush=True)
    else:
        print("\r{} [{}%] ".format(text, prog), end = "\r", flush=True)
        
def CountFileLines(filename):
    lines = 0
    with open(filename, 'rb') as file:
        buf_size = 1024 * 1024
        read_f = file.raw.read
        buf = read_f(buf_size)
        while buf:
            lines += buf.count(b'\n')
            buf = read_f(buf_size)
    return lines