import pandas as pd

import classifiers
import preprocess
import feature_eng

TRAIN_FILE = "../resources/dataFile/train-raw.tsv"
ALL_FILE = "../resources/dataFile/all-raw.tsv"
DEV_FILE = "../resources/dataFile/dev-raw.tsv"
TEST_FILE = "../resources/dataFile/test-raw.tsv"
PREDICT_PATH = "../resources/predict_results/"
FEATURE_PATH = "../resources/features/"

# classify based on TF-IDF and different feature engineering strategies
def tf_idf(threshold=0, length=0, limit=0, rm_unseen=False, rm_bias_word=True, merge=True, type="dev"):
    print("####INFO: start tf-idf")
    # read clean data and raw data from tsv (clean data finished preprocessing and substring extraction)
    if type == "dev":
        clean_train_x, train_y, train_id = preprocess.read_tsv(file_path=TRAIN_FILE, word_min_length=length)
        clean_test_x, test_y, test_id = preprocess.read_tsv(file_path=DEV_FILE, word_min_length=length)
        raw_test_x, test_y, test_id = preprocess.read_tsv(file_path=DEV_FILE, word_min_length=length, read_raw=True)
    else:
        clean_train_x, train_y, train_id = preprocess.read_tsv(file_path=ALL_FILE, word_min_length=length)
        clean_test_x, test_y, test_id = preprocess.read_tsv(file_path=TEST_FILE, word_min_length=length)
        raw_test_x, test_y, test_id = preprocess.read_tsv(file_path=TEST_FILE, word_min_length=length, read_raw=True)

    # feature engineering
    # replace unseen word in test set with most similar one in training base on Word2Vec
    if rm_unseen:
        clean_train_x, clean_test_x = feature_eng.replace_unknown_words(raw_train_x=clean_train_x, raw_test_x=clean_test_x)

    # eliminate bias word with low pcw
    if rm_bias_word:
        bias_words = feature_eng.reduce_words_pcw(df_x=clean_train_x, df_y=train_y, threshold=threshold, limit=limit)
        clean_train_x = feature_eng.filter_words(df_x=clean_train_x, bias_words=bias_words)
        clean_test_x = feature_eng.filter_words(df_x=clean_test_x, bias_words=bias_words)

    # merge all tweets of training/testing together
    if merge:
        clean_train_x, train_y = feature_eng.merge_train(clean_train_x, train_y)

    # extract tf-idf features
    train_x, test_x = feature_eng.tf_idf_extract(clean_train_x, clean_test_x, svd=False)

    # train / evaluate variant models
    if type == "dev":
        classifiers.evaluate(model_title=MODELS, train_x=train_x, train_y=train_y, test_x=test_x, test_y=test_y)
    elif type == "test":
        classifiers.predict(model_title=MODELS, train_x=train_x, train_y=train_y, test_x=test_x, raw_test_x=raw_test_x, test_id=test_id)
    
    print("####INFO: complete tf-idf")
    return

# classify based on Mutual Information
def mi(top="10", type="dev"):
    print("####INFO: start MI", top)
    # read features and row data
    train_file = FEATURE_PATH + "mi/" + "/train-top" + top + ".csv"
    test_file = FEATURE_PATH + "mi/" + "/" + type + "-top" + top + ".csv"
    if type == "dev":
        raw_test_x, test_y, test_id = preprocess.read_tsv(file_path=DEV_FILE, word_min_length=0, read_raw=True)
    else:
        raw_test_x, test_y, test_id = preprocess.read_tsv(file_path=TEST_FILE, word_min_length=0, read_raw=True)

    train_pd = pd.read_csv(train_file, delimiter=',')
    test_pd = pd.read_csv(test_file, delimiter=',')

    # allocate features
    train_id, train_x, train_y = train_pd.iloc[:, 0], train_pd.iloc[:, 1:-1], train_pd.iloc[:, -1]
    test_id, test_x, test_y = test_pd.iloc[:, 0], test_pd.iloc[:, 1:-1], test_pd.iloc[:, -1]

    # train / evaluate
    if type == "dev":
        classifiers.evaluate(model_title=MODELS, train_x=train_x, train_y=train_y, test_x=test_x, test_y=test_y)
    elif type == "test":
        classifiers.predict(model_title=MODELS, train_x=train_x, train_y=train_y, test_x=test_x, raw_test_x=raw_test_x, test_id=test_id)
    print("####INFO: complete MI")
    return

def wlh(top="10", length=0, type="dev"):
    print("####INFO: start WLH")
    # read clean data and raw data from tsv (clean data finished preprocessing and substring extraction)
    if type == "dev":
        clean_train_x, train_y, train_id = preprocess.read_tsv(file_path=TRAIN_FILE, word_min_length=length)
        clean_test_x, test_y, test_id = preprocess.read_tsv(file_path=DEV_FILE, word_min_length=length)
        raw_test_x, test_y, test_id = preprocess.read_tsv(file_path=DEV_FILE, word_min_length=length, read_raw=True)
    else:
        clean_train_x, train_y, train_id = preprocess.read_tsv(file_path=ALL_FILE, word_min_length=length)
        clean_test_x, test_y, test_id = preprocess.read_tsv(file_path=TEST_FILE, word_min_length=length)
        raw_test_x, test_y, test_id = preprocess.read_tsv(file_path=TEST_FILE, word_min_length=length, read_raw=True)

    train_x, test_x = feature_eng.extract_wlh(train_raw=clean_train_x, train_y=train_y, test_raw=clean_test_x, top=top)

    # train / evaluate variant models
    if type == "dev":
        classifiers.evaluate(model_title=MODELS, train_x=train_x, train_y=train_y, test_x=test_x, test_y=test_y)
    elif type == "test":
        classifiers.predict(model_title=MODELS, train_x=train_x, train_y=train_y, test_x=test_x, raw_test_x=raw_test_x,
                            test_id=test_id)
    print("####INFO: complete WLH")
    return

MODELS = ['MNB', 'EnsembleHard1']#['Dummy_most_frequent','Dummy_stratified','MNB','BMB','LinearSVM','LogisticRegression','EnsembleHard1','Stack']

# function calls
type = "dev"
'''
# MI
for top in ["10", "50", "100"]:
    mi(top=top, type=type)
'''
# WLH
#wlh(top="100", length=0, type=type)

# best MNB
#tf_idf(threshold=0.25, length=3, limit=0, rm_unseen=False, rm_bias_word=True, merge=True, type=type)
#tf_idf(threshold=0.25, length=3, limit=10, rm_unseen=True, rm_bias_word=True, merge=True, type=type)
# best Hard Ensemble1
tf_idf(threshold=0, length=0, limit=0, rm_unseen=False, rm_bias_word=False, merge=False, type=type)
# other test

#tf_idf(threshold=0.25, length=3, limit=10, rm_unseen=True, rm_bias_word=True, merge=True, type=type)
#tf_idf(threshold=0, length=0, limit=0, rm_unseen=False, rm_bias_word=False, merge=False, type=type)
