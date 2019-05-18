##########################################################################
# Short Text Location Prediction
#
# Project2 of course Machine Learning COMP30027 at Unimelb
#
# A simplification of the more general problem of geotagging, automatically identifies the location from which a textual
# message was sent, as one of four Australian cities
#
# Coordinator and Supervisor**: Tim Baldwin, Karin Verspoor, Jeremy Nicholson, Afshin Rahimi
# Python version: python3
# Author: Xiuge Chen
# Email: xiugec@student.unimelb.edu.au
# 2019.05.16
##########################################################################

import geotag_preprocess as preprocess
import geotag_feature_eng as feature
import geotag_classifiers as classifiers
import pandas as pd
import numpy as np
import time

#### Configuration ####
# preprocessing
STOP_WORDS = "rmStop"
#STOP_WORDS = "noRmStop"
#STEM = "stem"
STEM = "noStem"
# SPELL_CHECK = "checkSpell"
SPELL_CHECK = "noCheckSpell"

# feature extraction
EXTRACT_METHOD = "mi"
# EXTRACT_METHOD = "wlh"
# EXTRACT_METHOD = "ner"
TOP = "10"
# VECTORIZE = "embedding"
VECTORIZE = "noEmbedding"

#### Constant ####
TRAIN_FILE = "../resources/dataFile/train-raw.tsv"
DEV_FILE = "../resources/dataFile/dev-raw.tsv"
TEST_FILE = "../resources/dataFile/test-raw.tsv"

#### Function Declaration ####
# main entry
def geotag():
    raw_train, raw_dev, raw_test = read_tsv(TRAIN_FILE), read_tsv(DEV_FILE), read_tsv(TEST_FILE)

    clean_train = preprocess.preprocess(raw_train, stem=STEM, stop_words=STOP_WORDS, spell_check=SPELL_CHECK, type="train")
    clean_dev = preprocess.preprocess(raw_dev, stem=STEM, stop_words=STOP_WORDS, spell_check=SPELL_CHECK,
                                        type="dec")
    clean_test = preprocess.preprocess(raw_test, stem=STEM, stop_words=STOP_WORDS, spell_check=SPELL_CHECK,
                                        type="test")

    words_set, train_features = feature.extract_features(clean_train, reduce_method=EXTRACT_METHOD, top=TOP, vectorize=VECTORIZE,
                                                         type="train", words_set=None, tag=STEM+STOP_WORDS+SPELL_CHECK)
    words_set, dev_features = feature.extract_features(clean_dev, reduce_method=EXTRACT_METHOD, top=TOP, vectorize=VECTORIZE,
                                              type="dev", words_set=words_set, tag=STEM+STOP_WORDS+SPELL_CHECK)
    words_set, test_features = feature.extract_features(clean_test, reduce_method=EXTRACT_METHOD, top=TOP, vectorize=VECTORIZE,
                                              type="test", words_set=words_set, tag=STEM+STOP_WORDS+SPELL_CHECK)

    classifiers.train_n_test(train_df=train_features, dev_df=dev_features)
    classifiers.predict(train_df=train_features, test_df=test_features, test_original_df=raw_test, tags=STEM+STOP_WORDS+SPELL_CHECK+EXTRACT_METHOD+TOP+VECTORIZE)

    return

# read all content of tsv format file and return pandas dataframe
# Reason: use pandas to read tsv file (by set deliminator or sep) will accidentally miss rows whose previous row
# end up with \"", so to ensure data integrity, build a new function for reading tsv.
def read_tsv(file_path):
    print("####INFO: Start reading: ", file_path)
    data, count, num_lines, start_time = [], -1, sum(1 for line in open(file_path)), time.time()

    with open(file_path) as file:
        for line in file:
            words = line.rstrip().split('\t')

            row = [""] if count == -1 else [count]

            row.extend(words)
            data.append(row)
            count += 1

            print("####INFO: reading " + "{0:.0%}".format(count / num_lines), end='\r')

    data, end_time = np.array(data), time.time()

    print("####INFO: Complete reading:", file_path, "Spend Time:", end_time - start_time)

    return pd.DataFrame(data=data[1:,1:],
                  index=data[1:,0],
                  columns=data[0,1:])

#### Function Call ####
geotag()