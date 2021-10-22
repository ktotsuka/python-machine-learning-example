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

# Read movie review data
df = pd.read_csv('movie_data.csv', encoding='utf-8')

# Clean up the texts
df['review'] = df['review'].apply(preprocessor)

# Split data into training and testing sets
X_train = df.loc[:2500, 'review'].values # Use a small sample to reduce time for training the model (2500 instead of 25000)
y_train = df.loc[:2500, 'sentiment'].values
X_test = df.loc[25000:, 'review'].values
y_test = df.loc[25000:, 'sentiment'].values

# Set up a pipeline with TfidfTransformer (penalize words that appear in multiple documents) and logistic regression classifier
tfidf = TfidfVectorizer(strip_accents=None,
                        lowercase=False,
                        preprocessor=None)
lr_tfidf = Pipeline([('vect', tfidf),
                     ('clf', LogisticRegression(random_state=0, solver='liblinear'))])

# Set up parameters for grid search
nltk.download('stopwords')
stop = stopwords.words('english')
param_grid = [{'vect__ngram_range': [(1, 1)],
               'vect__stop_words': [stop, None],
               'vect__tokenizer': [tokenizer, tokenizer_porter],
               'clf__penalty': ['l1', 'l2'],
               'clf__C': [1.0, 10.0]},
              {'vect__ngram_range': [(1, 1)],
               'vect__stop_words': [stop, None],
               'vect__tokenizer': [tokenizer, tokenizer_porter],
               'vect__use_idf':[False],
               'vect__norm':[None],
               'clf__penalty': ['l1', 'l2'],
               'clf__C': [1.0, 10.0]},
              ]

# Perform the grid search to find out the best parameter set for the model and train the model with those parameter set
gs_lr_tfidf = GridSearchCV(lr_tfidf, param_grid,
                           scoring='accuracy',
                           cv=5, # number of folds for cross validation
                           verbose=2,
                           n_jobs=-1) # -1 for using all available processors at the same time
gs_lr_tfidf.fit(X_train, y_train) # This step takes a long time (~15 min)

# Print out the best parameter set and accuracy for the training data set
print('Best parameter set: %s ' % gs_lr_tfidf.best_params_)
print('CV Accuracy: %.3f' % gs_lr_tfidf.best_score_)

# Print out the accuracy for the testing data set
clf = gs_lr_tfidf.best_estimator_
print('Test Accuracy: %.3f' % clf.score(X_test, y_test))


