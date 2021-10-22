import os
import sys
import tarfile
import time
import urllib.request
import pyprind
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import re
from nltk.stem.porter import PorterStemmer
import nltk
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
import gzip
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from distutils.version import LooseVersion as Version
from sklearn import __version__ as sklearn_version
from sklearn.decomposition import LatentDirichletAllocation
from ch08lib import *

def stream_docs(path):
    with open(path, 'r', encoding='utf-8') as csv:
        next(csv)  # skip header
        for line in csv:
            text, label = line[:-3], int(line[-2])
            yield text, label

def get_minibatch(doc_stream, size):
    docs, y = [], []
    try:
        for _ in range(size):
            text, label = next(doc_stream)
            docs.append(text)
            y.append(label)
    except StopIteration:
        return None, None
    return docs, y

def tokenizer(text):
    text = preprocessor(text)
    stop = stopwords.words('english')
    tokenized = [w for w in text.split() if w not in stop]
    return tokenized

# Download stop words (and, are, is, etc.)
nltk.download('stopwords')

# Use HashingVectorizer instead of TfidfVectorizer because TfidfVectorizer cannot be used for out-of-core (incremental) learning
vect = HashingVectorizer(decode_error='ignore', 
                         n_features=2**21,
                         preprocessor=None, 
                         tokenizer=tokenizer)

# Use stochastic gradient descent classifier as the model (suitable for incremental learning)
clf = SGDClassifier(loss='log', random_state=1)

# Get stream for reading in the movie review data
doc_stream = stream_docs(path='movie_data.csv')

# Set up the progress bar (shown as ######...)
number_of_batches_for_training = 45
pbar = pyprind.ProgBar(number_of_batches_for_training)

# Fit the model incrementally 45 times with the batch size of 1000 samples
classes = np.array([0, 1])
for _ in range(number_of_batches_for_training):
    X_train, y_train = get_minibatch(doc_stream, size=1000)
    if not X_train:
        break
    X_train = vect.transform(X_train)
    clf.partial_fit(X_train, y_train, classes=classes)
    pbar.update()

# Get data from the testing data set
X_test, y_test = get_minibatch(doc_stream, size=5000)
X_test = vect.transform(X_test)

# Predict output for the testing data set and print the accuracy
print('Accuracy: %.3f' % clf.score(X_test, y_test))

# Use the testing data set to improve the model
clf = clf.partial_fit(X_test, y_test)

