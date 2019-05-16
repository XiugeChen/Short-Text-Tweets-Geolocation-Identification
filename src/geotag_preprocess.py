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

from nltk.tag import StanfordNERTagger
from nltk.tokenize import TweetTokenizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from spellchecker import SpellChecker
import pandas as pd
import numpy as np

#### Constant ####
# column names
TEXT_COL = "Text"
ID_COL = "Instance_ID"
LOC_COL = "Location"
# file paths
NER_MODEL_PATH = "../resources/stanford_ner/english.muc.7class.distsim.crf.ser.gz"
NER_SOURCE_PATH = "../resources/stanford_ner/stanford-ner.jar"


# preprocess each row of raw data, including
def preprocess(raw_data, word_reduce="ner"):
    print("####INFO: Start preprocessing data")

    cleaned_data = [[ID_COL, LOC_COL, TEXT_COL]]

    for index, row in raw_data.iterrows():
        text = row[TEXT_COL]

        correct_words = spell_check(tokenize_text(text))

        if word_reduce == "ner":
            result_words = stanford_ner(correct_words)
        elif word_reduce == "stop_list":
            result_words = remove_stop_words(correct_words)

        result_words = trace_stem(result_words)

        result_text = ''.join(word + "," for word in result_words)[:-1]

        new_row = [row[ID_COL], row[LOC_COL], result_text]
        cleaned_data.append(new_row)

    cleaned_data = np.array(cleaned_data)

    print("####INFO: Complete preprocessing data")
    return pd.DataFrame(data=cleaned_data[1:,1:],
                  index=cleaned_data[1:,0],
                  columns=cleaned_data[0,1:])

# Given a list of candidate words
# Return a list of LOCATION and ORGANIZATION words
def stanford_ner(words):
    st = StanfordNERTagger(NER_MODEL_PATH, NER_SOURCE_PATH)
    return [word for word, identifier in st.tag(words) if identifier in ["ORGANIZATION", "LOCATION"]]

# Given a list of candidate words
# Return a list of words that is neither stop words nor special words like url, emojis, mentions, hash tags...
def remove_stop_words(words):
    stop_words = set(stopwords.words('english'))
    return [word for word in words if word not in stop_words]

# Given a string of text from tweets
# Return a list of words after tokenize
def tokenize_text(text):
    tknzr = TweetTokenizer()
    return tknzr.tokenize(text)

# Use Peter Novig's method to find the words that may have been mis-spelled and try correct them
# Return the list of corrected words
def spell_check(words):
    spell = SpellChecker()
    return [spell.correction(word) for word in words]

# Given a list of words, use Porter stemmer to trace back to the stem of each words
# Return a list of stem words
def trace_stem(words):
    ps, stem_words = PorterStemmer(), []
    return [ps.stem(word) for word in words]