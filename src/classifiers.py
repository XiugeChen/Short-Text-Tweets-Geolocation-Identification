from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from mlxtend.classifier import StackingClassifier

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

import numpy as np
import time

PREDICT_PATH = "../resources/predict_results/"

# MODEL CONFIG
C_LG = 0.95
C_SVM = 0.95
max_iter = 10000

# MODEL SETTING
models = [MultinomialNB(),
          BernoulliNB(),
          svm.LinearSVC(C=C_SVM, max_iter=max_iter),
          LogisticRegression(solver="lbfgs", penalty="l2", C=C_LG, multi_class='auto', max_iter=max_iter),
          VotingClassifier(estimators=[('lr', LogisticRegression(solver="lbfgs", penalty='l2', C=C_LG, multi_class='auto', max_iter=max_iter)), ('svm', svm.LinearSVC(C=C_SVM, max_iter=max_iter)), ('mnb', MultinomialNB())], voting='hard'),
          VotingClassifier(estimators=[('lr', LogisticRegression(solver="lbfgs", penalty='l2', C=C_LG, multi_class='auto', max_iter=max_iter)), ('bmb', BernoulliNB()), ('mnb', MultinomialNB())], voting='hard'),
          VotingClassifier(estimators=[('lr', LogisticRegression(solver="lbfgs", penalty='l2', C=C_LG, multi_class='auto', max_iter=max_iter)), ('mnb', MultinomialNB()), ('bmb', BernoulliNB())], voting='soft', weights=[2, 2, 1]),
          StackingClassifier(classifiers=[MultinomialNB(), BernoulliNB()], meta_classifier=LogisticRegression(solver="lbfgs", penalty="l2", C=C_LG, max_iter=max_iter, multi_class='auto'), use_probas=True, average_probas=True),
          RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=1)]
titles = ['MNB',
          'BMB',
          'LinearSVM',
          'LogisticRegression',
          'EnsembleHard1',
          'EnsembleHard2',
          'EnsembleSoft1',
          'Stack',
          'RandomForest']

# evaluate training data on development data with variant classifiers
def evaluate(model_title, train_x, train_y, test_x, test_y):
    for title, model in zip(titles, models):
        if not title in model_title:
            continue

        print("####INFO: starting training", title)
        start = time.time()

        model.fit(train_x, train_y)
        acc = accuracy_score(model.predict(test_x), test_y)

        end = time.time()
        t = end - start
        print("####INFO: trainning", title, acc, 'time:', t)

    return

# output predict results of testing data
def predict(model_title, train_x, train_y, test_x, raw_test_x, test_id):
    for title, model in zip(titles, models):
        if not title in model_title:
            continue

        print("####INFO: starting predicting", title)
        start = time.time()

        model.fit(train_x, train_y)
        results = model.predict(test_x)
        tags = title + str(time.time())
        
        # output files
        file_path_txt = PREDICT_PATH + "submit-" + tags + ".txt"
        file_path_csv = PREDICT_PATH + "kaggle-" + tags + ".txt"
        write_file_txt = open(file_path_txt, 'w')
        write_file_csv = open(file_path_csv, 'w')
        write_file_csv.write("ID,Class\n")
        write_file_txt.write("ID | Text | Class\n")

        i = 0
        for result in results:
            id, text = test_id[i], raw_test_x[i]

            write_file_csv.write(str(id) + ',' + ''.join(i for i in result) + '\n')
            txt = str(id) + " | " + text + " | " + ''.join(i for i in result) + '\n'
            write_file_txt.write(txt)
            i += 1

        end = time.time()
        t = end - start
        print("####INFO: Predicting finish", title, 'time:', t)
    return

# iteratively predict results, means firstly predict only instances have prob higher than 
# threshold, then assume these instances are reliable, move them into training set to help 
# predict other instances until no instances could be predicted at that threshold, then
# lower down the threshold repeated previous step.
# if the threshold is lower than random guessing, get all unpredicted instances and guess
# them based on distribution
def iterative_predict_tfidf(raw_train_x, train_y, raw_test_x, test_y, model):
    print("####INFO: start iterative prediction")
    train_x_raw_copy, train_y_copy, test_x_raw_copy = list(raw_train_x.copy()), list(train_y.copy()), list(raw_test_x.copy())
    len_test_y = len(raw_test_x)
    test_predict_y = ["" for x in range(len_test_y)]

    stop_words = stopwords.words('english')
    cv = CountVectorizer(max_df=1.0, stop_words=stop_words, decode_error='ignore')
    transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
    
    # prediction distribution count
    mel, syd, bris, perth = 0, 0, 0, 0
    prob_t, left = 0.5, len_test_y
    # predict only instances have prob higher than threshold, then assume these instances 
    # are reliable, move them into training set to help predict other instances
    while prob_t >= 0.25:
        print("####INFO: iterative predict t", prob_t, left)
        cv_train_x = cv.fit_transform(train_x_raw_copy)
        cv_test_x = cv.transform(test_x_raw_copy)

        train_x = transformer.fit_transform(cv_train_x)
        test_x = transformer.transform(cv_test_x)

        model.fit(train_x, train_y_copy)

        find_predict = False
        for i in range(0, len_test_y):
            if not test_predict_y[i] == "":
                continue

            test_case = test_x[i]
            # get the heighest prob of prediction
            heighest_prob = max(model.predict_proba(test_case)[0])
            
            # if higher than threshold, trust it and move it to the training set
            if heighest_prob > prob_t:
                find_predict = True
                left -= 1
                label = model.predict(test_case)[0]
                test_predict_y[i] = label
                #if label != test_y[i]:
                #    print("#### WARNING: iterative predict incorrect: ", label, test_y[i], heighest_prob)

                if label == "Melbourne":
                    mel += 1
                elif label == "Brisbane":
                    bris += 1
                elif label == "Sydney":
                    syd += 1
                elif label == "Perth":
                    perth += 1

                train_x_raw_copy.append(test_x_raw_copy[i].copy())
                train_y_copy.append(label)

            print("####INFO: predicting " + "{0:.0%}".format(i / len_test_y), end='\r')

        if not find_predict:
            prob_t -= 0.001

    # predict those with no clue
    cv_train_x = cv.fit_transform(train_x_raw_copy)
    cv_test_x = cv.transform(test_x_raw_copy)

    train_x = transformer.fit_transform(cv_train_x)
    test_x = transformer.transform(cv_test_x)

    model.fit(train_x, train_y_copy)
    
    # randomly guess them based on the current distribution
    for i in range(0, len_test_y):
        if test_predict_y[i] == "":
            print("random guess:", i)
            test_case = test_x[i]
            heighest_prob = max(model.predict_proba(test_case)[0])

            if heighest_prob <= 0.2501:
                smallest = min([mel, syd, bris, perth])
                for i in range(0, 4):
                    if [mel, syd, bris, perth][i] == smallest:
                        if i == 0:
                            test_predict_y[i] = "Melbourne"
                            mel += 1
                        elif i == 1:
                            test_predict_y[i] = "Sydney"
                            syd += 1
                        elif i == 2:
                            test_predict_y[i] = "Brisbane"
                            bris += 1
                        elif i == 3:
                            test_predict_y[i] = "Perth"
                            perth += 1
            else:
                test_predict_y[i] = model.predict(test_case)[0]

            if test_predict_y[i] != test_y[i]:
                print("#### WARNING: predict incorrect: ", test_predict_y[i], test_y[i], heighest_prob)
    
    count = {}
    for predict_result in test_predict_y:
        if predict_result in count.keys():
            count[predict_result] += 1
        else:
            count[predict_result] = 1

    print("####INFO: final distribution:", count)

    return np.array(test_predict_y)