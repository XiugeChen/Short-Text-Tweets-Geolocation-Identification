[GloVe]: https://nlp.stanford.edu/projects/glove/

[CogComp]: https://github.com/CogComp/cogcomp-nlpy
[pyspellchecker]: https://pypi.org/project/pyspellchecker/



## Experiment Design

### Preprocessing

1. Remove all speical characters, url, emojis, mentions, hash tags and punctuations.

2. Spell check ([pyspellchecker], based on Peter Novig's method, Levenshtein Distance algorithm to find permutations)

4. Decrese words size:

   a. Remove all stop words. (SMART stop word list, 500 most common Twitter words) 

   b. NER (Standford or [CogComp])(ORGANIZATION/LOCATION) (char > 4 and frequency > 10) **(sometimes can't recognize ny, not work as examples)**

6. Porter stemming algorithm (also does reducing all chars to lowercase.) **(reduce melbourne -> melbourn, MEL -> mel)** OR Lemmatization

5. DBSCAN (*eps* = 0.5, *min_samples* = 5) OR K-means remove GLOBAL entities and entities far away from cluster

Not Do:

1. Exclude all Retweets. Reason: not worth 
2. Exclude all tweets being published by users whose cumulative words is less than 600 (?). Reason: not be able to do that.
3. Replace short forms(CA) / long forms(coooool) to unique full meaning. Reason: no library to do that and it is time-consuming to build a dictionary.

### Baseline

1. Dummy Classifier (Majority, Stratified, Uniform)

### Feature Engineering 

Options:

1. unigram(all words) / bigram (pair of words) / trigram (3 of words)
2. WLH (20% - 80%) on unigram(all words) / bigram (pair of words) / trigram (3 of words)
3. Word Embedding (Counted based [GloVe] / Prediction based word2vec / doc2vec)
4. WLH + Word Embedding  (?)
5. 

Not Consider:

1. Quadtree Data Partitioning: Not useful in this project since it solves the problem of unequally distributed data, which is not present in this project.

2. Paragraph Embeddings: TF-IDF(Term Frequency - Inverse Document Frequency)

   Normally consider a document a concatenation of tweets posted from the same user

   Not useful here since a word won't appear in the same tweet multiple times and we can't group tweets by users. 

### Classifiers

1. Multinomial Naive Bayes
2. Logistic Regression / Maximum Entropy
3. SVM (linear)
4. GMMs (?)
5. Ensemble Learning?
6. ANN / CNN / LSTM / NN: hard to interprete, long time to build and modify to best performance, lack of background and experience

Not Consider:

1. Semi-supervised: not suitable, since:

   a. number of unlabeled instances is not way more larger than labeled instances (roughly the same)

   b. unlabeled data is test data, train them into model would decrese the authenticity of test results (not well-represent general unseen data)