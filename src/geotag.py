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

#### Constant ####
TRAIN_FILE = "../resources/dataFile/train-raw.tsv"
DEV_FILE = "../resources/dataFile/dev-raw.tsv"
TEST_FILE = "../resources/dataFile/test-raw.tsv"

#### Function Declaration ####
# main entry
def geotag():
    raw_train = read_tsv(TRAIN_FILE)

    clean_train = preprocess.preprocess(raw_train, word_reduce="ner")

    #features = feature.extract_features(clean_train, "MI")

    return

# read all content of tsv format file and return pandas dataframe
# Reason: use pandas to read tsv file (by set deliminator or sep) will accidentally miss rows whose previous row
# end up with \"", so to ensure data integrity, build a new function for reading tsv.
def read_tsv(file_path):
    print("####INFO: Start reading: ", file_path)
    data, count = [], -1

    with open(file_path) as file:
        for line in file:
            line = line.rstrip()
            words = line.split('\t')

            if count == -1:
                row = [""]
            else:
                row = [count]

            row.extend(words)
            data.append(row)
            count += 1

    data = np.array(data)

    print("####INFO: Complete reading: ", file_path)

    return pd.DataFrame(data=data[1:,1:],
                  index=data[1:,0],
                  columns=data[0,1:])

#### Function Call ####
geotag()


