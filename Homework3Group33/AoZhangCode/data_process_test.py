"""
-----------------------------------------------------------------
CSI 5138: Assignment 3
Student:            Ao   Zhang
Student Number:     0300039680

-----------------------------------------------------------------
This code is for transferring all .txt file in the dataset into
one .txt file.

After that, feed the combined file into Glove to get the vocabulary
and the word vectors. Finally, all vocabulary and word vectors
are saved in the current directory. (vocab.txt, vectors.txt)
-----------------------------------------------------------------
"""
import re
import numpy as np
from glob import glob


def cleanSentences(string):
    """
    Function:
        Clean the sentances with these wierd characters.
    """
    strip_special_chars = re.compile("[^A-Za-z0-9 ]+")
    string = string.lower().replace("<br />", " ")
    return re.sub(strip_special_chars, "", string.lower())

# get all training data files
train_pos_files = glob("aclImdb/train/pos/*.txt")
train_neg_files = glob("aclImdb/train/neg/*.txt")
# get all test data files
test_pos_files = glob("aclImdb/test/pos/*.txt")
test_neg_files = glob("aclImdb/test/neg/*.txt")
# initialize the combined files
all_txt = None

# gather all trainining positive files into all_txt
for each_file in train_pos_files:
    f = open(each_file, "r")
    line=f.readline()
    cleaned_line = cleanSentences(line)
    if all_txt is None:
        all_txt = cleaned_line
    else:
        all_txt += " "
        all_txt += cleaned_line

# gather all trainining negative files into all_txt
for each_file in train_neg_files:
    f = open(each_file, "r")
    line=f.readline()
    cleaned_line = cleanSentences(line)
    if all_txt is None:
        all_txt = cleaned_line
    else:
        all_txt += " "
        all_txt += cleaned_line

# gather all test positive files into all_txt
for each_file in test_pos_files:
    f = open(each_file, "r")
    line=f.readline()
    cleaned_line = cleanSentences(line)
    if all_txt is None:
        all_txt = cleaned_line
    else:
        all_txt += "\n"
        all_txt += cleaned_line

# gather all test negative files into all_txt
for each_file in test_neg_files:
    f = open(each_file, "r")
    line=f.readline()
    cleaned_line = cleanSentences(line)
    if all_txt is None:
        all_txt = cleaned_line
    else:
        all_txt += " "
        all_txt += cleaned_line

# save all_txt as .txt file in order to feed into Glove
with open("aclimdb_data.txt", "w") as f:
    f.write(all_txt)