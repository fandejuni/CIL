# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 20:53:53 2019

@author: Justin
"""

from helper import PrintProgress, CountFileLines
from symspellpy.symspellpy import SymSpell, Verbosity
import pickle


def ReduceNumber(token):
    try:
        float(token)
        return "<0>"
    except:
        return token

def Dedouble(token):
    lc = token[0]
    r = ""
    n = 0
    for c in token+" ":
        if c != lc:
            r += lc*min(n,2)
            lc = c
            n = 0
        n += 1
    return r

def CleanToken(token):
    return Dedouble(ReduceNumber(token))

def CleanFile(filename, output_filename):
    num_lines = CountFileLines(filename)
    prog_text = "Cleaning file '{}'".format(filename)
    prog_threshold = num_lines/100
    PrintProgress(prog_text, 0)
    with open(output_filename, "w", encoding="utf8") as out_file:
        with open(filename, encoding="utf8") as file:
            for i, line in enumerate(file):
                if i>prog_threshold:
                    prog_threshold += num_lines/100
                    PrintProgress(prog_text, i*100//num_lines)
                cleanTokens = [CleanToken(token) for token in line.split()]
                cleanTokens = [token for token in cleanTokens if token != ""]
                out_file.write(" ".join(cleanTokens)+"\n")
    PrintProgress(prog_text, -1)
    
def UnnumberFile(filename, output_filename):
    num_lines = CountFileLines(filename)
    prog_text = "Unnumbering file '{}'".format(filename)
    prog_threshold = num_lines/100
    PrintProgress(prog_text, 0)
    with open(output_filename, "w", encoding="utf8") as out_file:
        with open(filename, encoding="utf8") as file:
            for i, line in enumerate(file):
                if i>prog_threshold:
                    prog_threshold += num_lines/100
                    PrintProgress(prog_text, i*100//num_lines)
                out_file.write(line.split(",",1)[1])
    PrintProgress(prog_text, -1)
    
def LoadSpellChecker(filename, update = False):
    base_filename, _ = filename.split(".")
    if not update:
        try:
            with open(base_filename+".pik", "rb") as file:
                print("Loading '{}.pik'... ".format(base_filename), end = "")
                sym_spell = pickle.load(file)
                print("[Done]")
        except:
            update = True
    if update:
        max_edit_distance_dictionary = 2
        prefix_length = 7
        sym_spell = SymSpell(max_edit_distance_dictionary, prefix_length)
        print("Loading '{}'... ".format(filename), end = "")
        if not sym_spell.load_dictionary(filename, 0, 1):
            print("\nDictionary file not found")
            return
        pickle.dump(sym_spell, open(base_filename+".pik", "wb" ))
        print("[Done]")
    return sym_spell

def SpellCheckToken(token, sym_spell, max_dist = 2):
    suggestion_verbosity = Verbosity.CLOSEST  # TOP, CLOSEST, ALL
    suggestions = sym_spell.lookup(token, suggestion_verbosity,
                                   max_dist)
    if len(suggestions) > 0:
        return suggestions[0].term
    return token

def SpellCheckFile(filename, output_filename, sym_spell, max_dist = 2):
    num_lines = CountFileLines(filename)
    prog_text = "Spellchecking file '{}'".format(filename)
    prog_threshold = num_lines/100
    PrintProgress(prog_text, 0)
    with open(output_filename, "w", encoding="utf8") as out_file:
        with open(filename, encoding="utf8") as file:
            for i, line in enumerate(file):
                if i>prog_threshold:
                    prog_threshold += num_lines/100
                    PrintProgress(prog_text, i*100//num_lines)
                spellCheckTokens = (SpellCheckToken(token, sym_spell, max_dist) 
                                    for token in line.split())
                out_file.write(" ".join(spellCheckTokens)+"\n")
    PrintProgress(prog_text, -1)