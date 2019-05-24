from gensim.models import Word2Vec
import numpy as np
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import OneHotEncoder
import time

#### Constant ####
# column names
TEXT_COL = "Text"
ID_COL = "Instance_ID"
LOC_COL = "Location"

# WLH constant
WLH_L = 2 # min length of word
WLH_C = 10 # overall count of a word

# merge train text from same place into same row
def merge_train(raw_train_x, train_y):

    mel, bri, syd, perth = "", "", "", ""
    for i in range(0, len(train_y)):
        if train_y[i] == "Melbourne":
            mel += raw_train_x[i] + " "
        elif train_y[i] == "Sydney":
            syd += raw_train_x[i] + " "
        elif train_y[i] == "Perth":
            perth += raw_train_x[i] + " "
        elif train_y[i] == "Brisbane":
            bri += raw_train_x[i] + " "
        else:
            print("wrong!!!!")

    return np.array([mel, bri, syd, perth]), np.array(["Melbourne", "Brisbane", "Sydney", "Perth"])

# reduce words size by removing words having low p(c|w), where c is city and w is word
def reduce_words_pcw(df_x, df_y, threshold, limit):
    words_count, city_count = {}, {}
    for i in range(0, len(df_x)):
        words, city = df_x[i].rstrip().split(' '), df_y[i]

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
    print("####INFO: finish count word")

    remove_words = set()
    # count p(c|w) for each word
    for word in words_count.keys():
        overall_count = 0.0
        # get total count
        for city in words_count[word].keys():
            overall_count += words_count[word][city]

        highest_score = 0
        # calculate score and store
        for city in words_count[word].keys():
            # p(c|w)
            score = words_count[word][city] / overall_count
            if score > highest_score:
                highest_score = score

        if abs(highest_score - 0.25) < threshold or overall_count < limit:
            remove_words.add(word)
            #print("eliminate:", word)
    print("####INFO: finish rule out word")

    return remove_words

# filter out words that don't show up in bias_words
def filter_words(df_x, bias_words):
    new_df_x = []

    for i in range(0, len(df_x)):
        words = df_x[i].rstrip().split(' ')

        new_row = []
        for word in words:
            if word not in bias_words:
                new_row.append(word)

        new_row = [word for word in words if word not in bias_words]

        new_str = ''.join(str(word) + ' ' for word in new_row)[:-1]
        new_df_x.append(new_str)
        print("####INFO: filtering " + "{0:.0%}".format(i / len(df_x)), end='\r')

    print("####INFO: finish apply to text")

    return np.array(new_df_x)

# replace unseen words in testing data by their most similar word in training data based on Word2Vec
def replace_unknown_words(raw_train_x, raw_test_x):
    new_test_x = raw_test_x.copy()

    words_set, words_list = set(), []
    # build word2vec model
    sentences = []
    for i in range(len(raw_train_x)):
        sentence = raw_train_x[i].rstrip().split(' ')
        sentences.append(sentence)

        words_list.extend(sentence)
        i += 1
        print("####INFO: counting " + "{0:.0%}".format(i / len(raw_train_x)), end='\r')

    words_set = set(words_list)
    
    for text in raw_test_x:
        sentence = text.rstrip().split(' ')
        sentences.append(sentence)

    word2vec_model = Word2Vec(sentences, min_count=1)

    # replace unseen word with similar word
    for i in range(len(raw_test_x)):
        words = raw_test_x[i].rstrip().split(' ')
        new_words = []

        for word in words:
            if word not in words_set:
                candidates = word2vec_model.wv.most_similar(word, topn=5)
                find = False
                for candidate, prob in candidates:
                    if candidate in words_set:
                        new_words.append(candidate)
                        find = True
                        break
                if find:
                    continue

            new_words.append(word)
        new_test_x[i] = ''.join(word + ' ' for word in new_words)[:-1]

        print("####INFO: replacing " + "{0:.0%}".format(i / len(raw_test_x)), end='\r')
    return raw_train_x, new_test_x

# extract tf-idf features
def tf_idf_extract(clean_train_x, clean_test_x, svd=False):
    stop_words = stopwords.words('english')

    cv = CountVectorizer(max_df=1.0, stop_words=stop_words, decode_error='ignore')
    transformer = TfidfTransformer(smooth_idf=True, use_idf=True)

    cv_train_x = cv.fit_transform(clean_train_x)
    cv_test_x = cv.transform(clean_test_x)

    train_x = transformer.fit_transform(cv_train_x)
    test_x = transformer.transform(cv_test_x)
    
    # reduce dimension if required
    if svd:
        pca = TruncatedSVD(n_components=1000)
        train_x = pca.fit_transform(train_x)
        test_x = pca.transform(test_x)

    return train_x, test_x

# select features base on chi2 or other critiria
def select_features(train_x, train_y):
    new_train_x = SelectPercentile(chi2, percentile=30).fit_transform(train_x, train_y)
    return new_train_x

# extract a limited set (specified by top) of words from the dataframe based on WLH and construct feature vectors
def extract_wlh(train_raw, train_y, test_raw, top):
    print("####INFO: start extracing WLH top:", top)
    start_time = time.time()
    # count the P(w|c) and P(w), where w means word and c means city
    words_count, total_words = {}, 0

    for i in range(len(train_raw)):
        words, city = train_raw[i].rstrip().split(' '), train_y[i]
        # count words and their individual count in cities
        for word in words:
            if len(word) < WLH_L:
                continue

            word = word.lower()
            if word in words_count:
                if city in words_count[word]:
                    words_count[word][city] += 1
                else:
                    words_count[word][city] = 1
            else:
                words_count[word] = {}
                words_count[word][city] = 1
                total_words += 1

    wlh_all = {}
    # count p(c|w) for each word
    for word in words_count.keys():
        overall_count = 0.0
        # get total count
        for city in ["Melbourne", "Brisbane", "Sydney", "Perth"]:
            if city not in words_count[word].keys():
                words_count[word][city] = 0
            else:
                overall_count += words_count[word][city]

        if overall_count < WLH_C:
            continue

        highest_score = 0
        # calculate score and store
        for city in words_count[word].keys():
            # p(c|w)
            pcw = words_count[word][city] / overall_count

            if pcw > highest_score:
                highest_score = pcw

        if highest_score >= 0.5:
            wlh_all[word] = highest_score

    # rank the word by WLH
    wlh_all_sorted_keys = sorted(wlh_all, key=wlh_all.get, reverse=True)
    limit, count = (float(top) / 100) * len(wlh_all.keys()), 0
    new_word = []
    for word in wlh_all_sorted_keys:
        new_word.append(word)
        count += 1
        if count > limit:
            break

    train_x_transform = [text.rstrip().split(' ') for text in train_raw]
    test_x_transform = [text.rstrip().split(' ') for text in test_raw]

    length = len(new_word)
    train_x_transform = np.array([xi + [None] * (length - len(xi)) for xi in train_x_transform])
    test_x_transform = np.array([xi + [None] * (length - len(xi)) for xi in test_x_transform])
    new_word = np.array([new_word])

    # do one hot encoding
    onehot_encoder = OneHotEncoder(sparse=True, handle_unknown='ignore')
    onehot_encoder.fit(new_word)
    new_train_features = onehot_encoder.transform(train_x_transform).toarray()
    new_test_features = onehot_encoder.transform(test_x_transform).toarray()

    print("####INFO: Complete extracing WLH, time", str(time.time() - start_time))
    return new_train_features, new_test_features