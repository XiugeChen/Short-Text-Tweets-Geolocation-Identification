from nltk.tokenize import TweetTokenizer
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
import numpy as np
import time
import re

# read tsv datafile
def read_tsv(file_path, remove_empty=True, word_min_length=0, read_raw=False):
    print("####INFO: Start reading: ", file_path)
    start_time = time.time()

    x, y, id = [], [], []
    with open(file_path) as file:
        for line in file:
            texts = line.rstrip().split('\t')

            if read_raw:
                x.append(texts[-1])
            else:
                words = extract_word(line=texts[-1].lower(), min_length=word_min_length)

                content = ''.join(word + ' ' for word in words)
                # remove empty
                if remove_empty and not content == ' ' and not content == '':
                    x.append(content)
                else:
                    x.append(content)

            id.append(texts[0])
            y.append(texts[1])

    end_time = time.time()
    print("####INFO: Complete reading:", file_path, "Spend Time:", end_time - start_time)
    return np.array(x), np.array(y), np.array(id)

# extract words from a line, tokenize line first, eliminate words with length smaller than
# min_length, than break words down into substrings with length 6
def extract_word(line, min_length):
    words = tokenize_and_filter_word(line)

    # also break word into substring, best performance len = 6, 0.35108264551398866
    SUB = 6
    new_words = []
    for word in words:
        if len(word) < min_length:
            continue

        if len(word) > SUB:
            for i in range(0, len(word) - SUB):
                sub_word = word[i:i + SUB]
                new_words.append(sub_word)
        else:
            new_words.append(word)

    # words = remove_stop_words(words)
    # words = trace_stem(words)
    return new_words

# Given a list of candidate words
# Return a list of words do not contain stop words
def remove_stop_words(words):
    stop_words = set(stopwords.words('english'))
    return [word for word in words if word not in stop_words]

# Given a string of text from tweets, remove words contain special words like url, emojis, mentions, numbers, hash tags... and tokenize them
# Return a list of words after tokenize
def tokenize_and_filter_word(text):
    tknzr = TweetTokenizer()

    text = re.sub(r'\\u[A-Za-z0-9]{4}', '', text)
    # text = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", text).split())
    text = ' '.join(re.sub("([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", text).split())
    text = re.sub("\d+", " ", text)

    return tknzr.tokenize(text)

# Given a list of words, use Snowball stemmer to trace back to the stem of each words
# Return a list of stem words
def trace_stem(words):
    stemmer = SnowballStemmer("english")
    return [stemmer.stem(word) for word in words]