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

# Read the movie review data
df = pd.read_csv('movie_data.csv', encoding='utf-8')

# Vectorize (use words as features) the reviews
count = CountVectorizer(stop_words='english', # use scikit-learn's built-in English stop-word library
                        max_df=.1, # ignore words that appear in more than 10% of the documents
                        max_features=5000) # maximum number of features (words) 
X = count.fit_transform(df['review'].values)

# Classify the review into topics
lda = LatentDirichletAllocation(n_components=10, # number of topics
                                random_state=123,
                                learning_method='batch')
X_topics = lda.fit_transform(X) # This step takes a long time (~10min).  X_topics contains the strength for each topic for each review

# Print the top 5 words for each topic
n_top_words = 5
feature_names = count.get_feature_names() # It contains the 5000 features (words)
for topic_idx, topic in enumerate(lda.components_):
    print("Topic %d:" % (topic_idx + 1))
    # "topic" contains the feature strength for each of 5000 features
    # argsort() will provide the indeces in order of feature strength (weak to strong)
    # to get the 5 strongest feature, you need to grab the last 5 in reverse order
    print(" ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))

# Based on reading the 5 most important words for each topic, we may guess that the LDA identified the following topics:
#     
# 1. Generally bad movies (not really a topic category)
# 2. Movies about families
# 3. War movies
# 4. Art movies
# 5. Crime movies
# 6. Horror movies
# 7. Comedies
# 8. Movies somehow related to TV shows
# 9. Movies based on books
# 10. Action movies

# To confirm that the categories make sense based on the reviews, let's plot 5 movies from the horror movie category (category 6 at index position 5):
horror = X_topics[:, 5].argsort()[::-1] # contains the strength for "horror" topic for each review
for iter_idx, movie_idx in enumerate(horror[:3]):
    print('\nHorror movie #%d:' % (iter_idx + 1))
    print(df['review'][movie_idx][:300], '...')
