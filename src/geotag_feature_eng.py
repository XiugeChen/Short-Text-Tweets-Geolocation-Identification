##########################################################################
# Short Text Location Prediction
#
# Project2 of course Machine Learning COMP30027 at Unimelb
#
# Provide tweet text feature engineering, including
#
# Coordinator and Supervisor**: Tim Baldwin, Karin Verspoor, Jeremy Nicholson, Afshin Rahimi
# Python version: python3
# Author: Xiuge Chen
# Email: xiugec@student.unimelb.edu.au
# 2019.05.16
##########################################################################

from nltk.tag import StanfordNERTagger
import pandas as pd
import numpy as np
import os
import time

#### Constant ####
# column names
TEXT_COL = "Text"
ID_COL = "Instance_ID"
LOC_COL = "Location"

# file paths
FEATURE_PATH = "../resources/features/"
NER_MODEL_PATH = "../resources/stanford_ner/english.muc.7class.distsim.crf.ser.gz"
NER_SOURCE_PATH = "../resources/stanford_ner/stanford-ner.jar"

def extract_features(dataframe, reduce_method="mi", top="10", vectorize="no", type="train", tag=""):
    print("####INFO: Start extracting features")
    start_time = time.time()

    # further decrease the size of words
    if reduce_method == "mi":
        features = extract_mi(dataframe, top, type, tag)
    elif reduce_method == "ner":
        features = extract_ner(dataframe, top, type, tag)
    elif reduce_method == "wlh":
        features = extract_wlh(dataframe, top, type, tag)
    else:
        features = extract_all(dataframe, top, type, tag)

    if vectorize == "embedding":
        features = vectorize(features)

    end_time = time.time()
    print("####INFO: Complete extracting features", "Spend Time:", end_time - start_time)

    return features

# extract limited set (specified by top) of words from the dataframe based on Mutual Information and construct feature vectors
# return a dataframe of features
def extract_mi(dataframe, top, type, tag):
    if top in ["10", "100", "50"] and type in ["train", "test", "dev"]:
        file = FEATURE_PATH + "mi/" + "/" + type + "-top" + top + "-" + tag + ".csv"
        return pd.read_csv(file, delimiter=',')
    else:
        return None

# extract all NER words from the dataframe and construct feature vectors
def extract_ner(dataframe, type, tag):
    file_path = FEATURE_PATH + "ner/" + type + "-top" + top + "-" + tag + ".csv"
    if os.path.isfile(file_path):
        return pd.read_csv(file_path, delimiter=',')

    words_set = set()
    for index, row in dataframe.iterrows():
        words = row[TEXT_COL].split(',')
        new_words = stanford_ner(words)
        words_set = words_set.union(set(new_words))

    return build_feature_df(dataframe, words_set)

# extract a limited set (specified by top) of words from the dataframe based on WLH and construct feature vectors
def extract_wlh(dataframe, top, type, tag):
    file_path = FEATURE_PATH + "wlh/" + type + "-top" + top + "-" + tag + ".csv"
    if os.path.isfile(file_path):
        return pd.read_csv(file_path, delimiter=',')

    # count the P(w|c) and P(w), where w means word and c means city
    words_count, city_count,total_cities, total_words = {}, {}, 0, 0
    for index, row in dataframe.iterrows():
        words, city = row[TEXT_COL].split(','), row[LOC_COL]

        # count total cities
        if city in city_count:
            city_count[city] += 1
        else:
            city_count[city] = 1
        total_cities += 1

        # count words and their individual count in cities
        for word in words:
            if word in words_count:
                if city in words_count[word]:
                    words_count[word][city] += 1
                else:
                    words_count[word][city] = 1
            else:
                words_count[word] = {}
                words_count[word][city] = 1
                total_words += 1

    # count WLH for each word
    wlh = {}
    for word in words_count.keys():
        highest_pwc, overall_count = 0.0, 0.0
        for city in words_count[word].keys():
            overall_count += words_count[word][city]

            pwc = words_count[word][city] / city_count[city]
            if (pwc > highest_pwc):
                highest_pwc = pwc
        pw = overall_count / total_cities

        wlh[word] = highest_pwc / pw

    # rank the word by WLH
    wlh_sorted_keys = sorted(wlh, key=wlh.get, reverse=True)

    # only include top percent of words into features
    limit, count, words_set = int(top) / 100, 0, set()
    for word in wlh_sorted_keys:
        words_set.add(word)
        if (count > limit):
            break

    return build_feature_df(dataframe, words_set)

# extract all words from the dataframe and construct feature vectors
def extract_all(dataframe, top, type, tag):
    file_path = FEATURE_PATH + "all/" + type + "-top" + top + "-" + tag + ".csv"
    if os.path.isfile(file_path):
        return pd.read_csv(file_path, delimiter=',')

    words_set = set()
    for index, row in dataframe.iterrows():
        words = row[TEXT_COL].split(',')
        words_set = words_set.union(set(words))

    return build_feature_df(dataframe, words_set)

# Given a list of candidate words
# Return a list of LOCATION and ORGANIZATION words
def stanford_ner(words):
    st = StanfordNERTagger(NER_MODEL_PATH, NER_SOURCE_PATH)
    return [word for word, identifier in st.tag(words) if identifier in ["ORGANIZATION", "LOCATION"]]

# Given a dataframe and all chosen words
# Return a dataframe of features
def build_feature_df(dataframe, words_set):
    feature_list, header, count, map, len_words = [], [ID_COL, LOC_COL], 2, {}, len(words_set)

    for word in words_set:
        header.append(word)
        map[word] = count
        count += 1

    feature_list.append(header)

    for index, row in dataframe.iterrows():
        new_row = [row[ID_COL], row[LOC_COL]]
        new_row.extend([0] * len_words)

        words = row[TEXT_COL]
        for word in words:
            new_row[map[word]] = 1

        feature_list.append(new_row)

    feature_array = np.array(feature_list)

    return pd.DataFrame(data=feature_array[1:, 1:],
                        index=feature_array[1:, 0],
                        columns=feature_array[0, 1:])


def vectorize(feature_list):

    return None