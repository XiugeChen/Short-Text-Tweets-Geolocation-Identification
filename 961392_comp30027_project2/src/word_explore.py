import pandas as pd

# column names
TEXT_COL = "Text"
ID_COL = "Instance_ID"
LOC_COL = "Location"

# WLH constant
WLH_L = 2

TRAIN_FILE = "../resources/preprocess_data/train_stem_rmStop.csv"
DEV_FILE = "../resources/preprocess_data/dev_stem_rmStop.csv"
TEST_FILE = "../resources/preprocess_data/test_stem_rmStop.csv"
OUTPUT = "../resources/data_exploration/"

def explore_data():
    print("####INFO: Start reading data")
    train_df = pd.read_csv(TRAIN_FILE, delimiter=',')
    dev_df = pd.read_csv(DEV_FILE, delimiter=',')
    test_df = pd.read_csv(TEST_FILE, delimiter=',')
    print("####INFO: Complete reading data")

    # file to see the distribution of words
    inAllThree = open(OUTPUT+"inAllThree.txt", 'w')
    trainButNotTest = open(OUTPUT+"trainButNotTest.txt", 'w')
    testButNotTrain = open(OUTPUT+"testButNotTrain.txt", 'w')
    devButNotTrain = open(OUTPUT + "devButNotTrain.txt", 'w')

    train_words_count, train_pcw, train_sorted_keys = count_word(train_df)
    dev_words_count, dev_pcw, dev_sorted_keys = count_word(dev_df)
    test_words_count, test_pcw, test_sorted_keys = count_word(test_df)

    # find out which popular word is not in train or test
    print("####INFO: Start go through words")
    for key in train_sorted_keys:
        if key not in dev_sorted_keys or key not in test_sorted_keys:
            trainButNotTest.write(str(key) + "\t" + str(train_words_count[key]) + "\n")
        else:
            inAllThree.write(str(key) + "\t" + str(train_words_count[key]) + "\t" + str(dev_words_count[key]) + "\t" + str(test_words_count[key]) + "\n")

    for key in test_sorted_keys:
        if key not in train_words_count:
            testButNotTrain.write(str(key) + "\t" + str(test_words_count[key]) + "\n")

    for key in dev_sorted_keys:
        if key not in train_words_count:
            devButNotTrain.write(str(key) + "\t" + str(dev_words_count[key]) + "\n")

    # find out which sentences in test doesn't have any of word appeared in trainning set
    test_sen_not_in_train = open(OUTPUT + "test_sen_not_in_train.txt", 'w')
    for index, row in test_df.iterrows():
        words = str(row[TEXT_COL]).split(',')
        find = False
        for word in words:
            if word in train_sorted_keys:
                find = True
                break

        if not find:
            test_sen_not_in_train.write(''.join(word + ',' for word in words) + "\n")

    # find out which sentences in dev doesn't have any of word appeared in trainning set
    dev_sen_not_in_train = open(OUTPUT + "dev_sen_not_in_train.txt", 'w')
    for index, row in dev_df.iterrows():
        words = str(row[TEXT_COL]).split(',')
        find = False
        for word in words:
            if word in train_sorted_keys:
                find = True
                break

        if not find:
            dev_sen_not_in_train.write(''.join(word + ',' for word in words) + "\n")

    print("####INFO: Complete go through words")
    return

# count the words distribution, corresponding p(c|w), the city each word appeared most frequent, and a sorted list of
# word based on their p(c|w) decendingly
def count_word(df):
    print("####INFO: Start Count data")
    # count the P(w|c) and P(w), where w means word and c means city
    words_count = {}
    for index, row in df.iterrows():
        words, city = str(row[TEXT_COL]).split(','), row[LOC_COL]

        # count words and their individual count in cities
        for word in words:
            word = word.lower()
            if word in words_count:
                if city in words_count[word]:
                    words_count[word][city] += 1
                else:
                    words_count[word][city] = 1
            else:
                words_count[word] = {}
                words_count[word][city] = 1

    # count p(c|w) for each word
    pcw, high_city, overall = {}, {}, {}
    for word in words_count.keys():
        overall_count = 0.0
        # get total count
        for city in ["Melbourne", "Brisbane", "Sydney", "Perth", "?"]:
            if city not in words_count[word].keys():
                words_count[word][city] = 0
            else:
                overall_count += words_count[word][city]

        overall[word] = overall_count

        highest_score, highest_city = 0, "?"
        # calculate score and store
        for city in words_count[word].keys():
            # p(c|w)
            if overall_count == 0 or words_count[word][city] == 0:
                score = 0
            else:
                score = words_count[word][city] / overall_count

            if score > highest_score:
                highest_score = score
                highest_city = city

        pcw[word] = highest_score
        high_city[word] = highest_city

    sorted_keys = sorted(overall, key=overall.get, reverse=True)

    print("####INFO: Complete Count data")
    return words_count, pcw, high_city, sorted_keys

explore_data()