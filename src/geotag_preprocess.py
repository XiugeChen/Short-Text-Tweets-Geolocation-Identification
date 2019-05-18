##########################################################################
# Short Text Location Prediction
#
# Project2 of course Machine Learning COMP30027 at Unimelb
#
# Provide tweet text preprocessing, including
#
# Coordinator and Supervisor**: Tim Baldwin, Karin Verspoor, Jeremy Nicholson, Afshin Rahimi
# Python version: python3
# Author: Xiuge Chen
# Email: xiugec@student.unimelb.edu.au
# 2019.05.16
##########################################################################

from nltk.tokenize import TweetTokenizer
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from spellchecker import SpellChecker
import pandas as pd
import numpy as np
import os
import time
import re

#### Constant ####
# column names
TEXT_COL = "Text"
ID_COL = "Instance_ID"
LOC_COL = "Location"
# file paths
PREPROCESS_PATH = "../resources/preprocess_data/"

# preprocess each row of raw data, including
def preprocess(raw_dataframe, stem="stem", stop_words="rmStop", spell_check="checkSpell", type="train"):
    print("####INFO: Start preprocessing data")

    start_time = time.time()

    # if preprocess is done previously, just read the results
    file_path = PREPROCESS_PATH + type + "_" + stem + "_" + stop_words + "_" + spell_check + ".csv"
    if os.path.isfile(file_path):
        end_time = time.time()
        print("####INFO: Complete preprocessing data (Read from existing)", "Spend Time:", end_time - start_time)
        return pd.read_csv(file_path, delimiter=',')

    cleaned_data, new_words = [[ID_COL, LOC_COL, TEXT_COL]], set()
    write_file = open(file_path, 'a')
    write_file.write(ID_COL + "," + LOC_COL + "," + TEXT_COL + '\n')

    for index, row in raw_dataframe.iterrows():
        text = row[TEXT_COL]

        # preprocessing
        correct_words = tokenize_text(text)
        result_words = correct_words

        if spell_check == "checkSpell":
            result_words = spell_check(result_words)

        if stop_words == "rmStop":
            result_words = remove_stop_words(result_words)

        if stem == "stem":
            result_words = trace_stem(result_words)

        # count the impact of preprocessing
        new_words = new_words.union(set(result_words))

        # output format preprocessing data
        result_text = ''.join(word + "," for word in result_words)[:-1]

        new_row = [row[ID_COL], row[LOC_COL], result_text]
        cleaned_data.append(new_row)

        # store it in file for next reading
        row_csv = new_row[0] + "," + new_row[1] + ",\"" + new_row[2] + '\"\n'
        write_file.write(row_csv)

        #print(row_csv)
        print("####INFO: processing " + "{0:.0%}".format(float(index) / float(raw_dataframe.shape[0])), end='\r')

    cleaned_data, end_time = np.array(cleaned_data), time.time()

    print("####INFO: Complete preprocessing data", "Spend Time:", end_time - start_time)
    print("####INFO: Reduce words set length of " + len(new_words))

    return pd.DataFrame(data=cleaned_data[1:,1:],
                  index=cleaned_data[1:,0],
                  columns=cleaned_data[0,1:])

# Given a list of candidate words
# Return a list of words do not contain stop words
def remove_stop_words(words):
    stop_words = set(stopwords.words('english'))
    return [word for word in words if word not in stop_words]

# Given a string of text from tweets, remove words contain special words like url, emojis, mentions, hash tags... and tokenize them
# Return a list of words after tokenize
def tokenize_text(text):
    tknzr = TweetTokenizer()

    text = re.sub(r'\\u[A-Za-z0-9]{4}', '', text)
    text = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", text).split())

    return tknzr.tokenize(text)

# Use Peter Novig's method to find the words that may have been mis-spelled and try correct them
# Return the list of corrected words
def spell_check(words):
    spell = SpellChecker()
    return [spell.correction(word) for word in words]

# Given a list of words, use Snowball stemmer to trace back to the stem of each words
# Return a list of stem words
def trace_stem(words):
    stemmer = SnowballStemmer("english")
    return [stemmer.stem(word) for word in words]