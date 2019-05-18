##########################################################################
# Short Text Location Prediction
#
# Project2 of course Machine Learning COMP30027 at Unimelb
#
# Provide classifiers for trainning and testing
#
# Coordinator and Supervisor**: Tim Baldwin, Karin Verspoor, Jeremy Nicholson, Afshin Rahimi
# Python version: python3
# Author: Xiuge Chen
# Email: xiugec@student.unimelb.edu.au
# 2019.05.18
##########################################################################

from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import time
import numpy as np

#### Constant ####
# column names
TEXT_COL = "Text"
ID_COL = "Instance_ID"
LOC_COL = "Location"

# MODEL SETTING
models = [DummyClassifier(strategy='most_frequent'),
          DummyClassifier(strategy='stratified'),
          MultinomialNB(),
          svm.LinearSVC(max_iter=10000),
          LogisticRegression(solver="saga", max_iter=1000)]
titles = ['Dummy_most_frequent',
          'Dummy_stratified',
          'MNB',
          'LinearSVC',
          'LogisticRegression']

# file paths
PREDICT_PATH = "../resources/predict_results/"

def train_n_test(train_df, dev_df):
    train_features = list(set(train_df.columns.values.tolist()) - set([ID_COL, LOC_COL]))
    dev_features = list(set(dev_df.columns.values.tolist()) - set([ID_COL, LOC_COL]))

    x_train, y_train, x_test, y_test = train_df.loc[:, train_features], train_df.loc[:, LOC_COL], dev_df.loc[:, dev_features], dev_df.loc[:, LOC_COL]

    for title, model in zip(titles, models):
        start = time.time()
        model.fit(x_train, y_train)
        acc = accuracy_score(model.predict(x_test), y_test)
        end = time.time()
        t = end - start
        print("####INFO: Trainning", title, acc, 'time:', t)

    return

def predict(train_df, test_df, test_original_df, tags):
    train_features = list(set(train_df.columns.values.tolist()) - set([ID_COL, LOC_COL]))
    test_features = list(set(test_df.columns.values.tolist()) - set([ID_COL, LOC_COL]))

    x_train, y_train, x_test = train_df.loc[:, train_features], train_df.loc[:, LOC_COL], test_df.loc[:, test_features]

    for title, model in zip(titles, models):
        start = time.time()
        model.fit(x_train, y_train)
        results = model.predict(x_test)
        file_path = PREDICT_PATH + title + "_predict" + "_" + tags + ".txt"
        write_file = open(file_path, 'w')

        i = 0
        for result in results:
            row = test_original_df.iloc[i, :]
            id, text = row[ID_COL], row[TEXT_COL]

            txt = str(id) + " | " + text + " | " + ''.join(i for i in result) + '\n'
            write_file.write(txt)

            i += 1

        end = time.time()
        t = end - start
        print("####INFO: Predicting", title, 'time:', t)

    return