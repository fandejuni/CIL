# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 19:27:58 2019

@author: Justin
"""

from cleaner import CleanFile, LoadSpellChecker, SpellCheckFile, UnnumberFile
from dict_file_manip import CreateDictFromFile, AddDictFiles

CleanFile("twitter-datasets/train_pos.txt", "processed/train_pos_clean.txt")
CleanFile("twitter-datasets/train_neg.txt", "processed/train_neg_clean.txt")

CreateDictFromFile("processed/train_pos_clean.txt", "processed/dict_pos.txt", 50)
CreateDictFromFile("processed/train_neg_clean.txt", "processed/dict_neg.txt", 50)

AddDictFiles(["processed/dict_pos.txt", "processed/dict_neg.txt"], "processed/dict.txt")

sym_spell = LoadSpellChecker("processed/dict.txt")

SpellCheckFile("processed/train_pos_clean.txt", "processed/train_pos_spellchecked.txt", sym_spell)
SpellCheckFile("processed/train_neg_clean.txt", "processed/train_neg_spellchecked.txt", sym_spell)

UnnumberFile("twitter-datasets/test_data.txt", "processed/test_unnumbered.txt")
CleanFile("processed/test_unnumbered.txt", "processed/test_clean.txt")
SpellCheckFile("processed/test_clean.txt", "processed/test_spellchecked.txt", sym_spell)

