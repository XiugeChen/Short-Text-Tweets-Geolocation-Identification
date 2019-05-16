[GloVe]: https://nlp.stanford.edu/projects/glove/

[CogComp]: https://github.com/CogComp/cogcomp-nlpy



## Experiment Design

### Preprocessing

1. Exclude all (generic?) hash tags and mentions
2. Exclude all Retweets (?)
3. Remove all speical characters, url, emojis and punctuations.
4. Reduce all chars to lowercase.
5. Spell check (?) (replace according to Jaccard coefficient for strings)
6. Porter stemming algorithm
7. Remove all stop words. (SMART stop word list, 500 most common Twitter words) OR NER (Standford or [CogComp])(ORGANIZATION/LOCATION) (char > 4 and frequency > 10(?))
8. DBSCAN (*eps* = 0.5, *min_samples* = 5) OR K-means remove GLOBAL entities and entities far away from cluster
9. Exclude all tweets being published by users whose cumulative words is less than 600 (?)
10. Tweet Normalization (transfer short forms(CA) / long forms(coooool) to full meanings) (?)

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