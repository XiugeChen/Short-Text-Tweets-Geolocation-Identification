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
import pandas as pd
import numpy as np
import time

#### Configuration ####
# preprocessing
STOP_WORDS = "rmStop"
#STOP_WORDS = "noRmStop"
STEM = "stem"
# STEM = "noStem"

# feature extraction
EXTRACT_METHOD = "mi"
# EXTRACT_METHOD = "wlh"
# EXTRACT_METHOD = "ner"
# VECTORIZE = "embedding"

#### Constant ####
TRAIN_FILE = "../resources/dataFile/train-raw.tsv"
DEV_FILE = "../resources/dataFile/dev-raw.tsv"
TEST_FILE = "../resources/dataFile/test-raw.tsv"

#### Function Declaration ####
# main entry
def geotag():
    raw_train = read_tsv(TRAIN_FILE)

    clean_train = preprocess.preprocess(raw_train, stem=STEM, stop_words=STOP_WORDS, type="train")

    #features = feature.extract_features(clean_train, reduce_method="mi", top="10", vectorize="embedding", type="train", tag=STEM + STOP_WORDS)
    # if features == None:
    #    return

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