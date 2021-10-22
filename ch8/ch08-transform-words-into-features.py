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

# Create a sample sentences
docs = np.array([
        'The sun is shining',
        'The weather is sweet',
        'The sun is shining, the weather is sweet, and one and one is two'])

# Vectorize the sentences with CountVectorizer
count = CountVectorizer()
bag = count.fit_transform(docs)
print(count.vocabulary_) # Print the list of vocabulary found and thier assigned feature index (index is alphabetical order)
print(bag.toarray()) # Print the features (each feature is the number of times a particular word occurs in a sentence)

# Transform the vectors with TfidfTransformer (penalize words that appear in multiple documents)
tfidf = TfidfTransformer(use_idf=True, 
                         norm='l2', # normalize the vector with L2-normalization
                         smooth_idf=True)
np.set_printoptions(precision=2)
print(tfidf.fit_transform(bag).toarray())

