import pickle
import os
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from tokenizer import *

# Download stopwords (are, is, and, etc.)
nltk.download('stopwords')
stop = stopwords.words('english')

# Read the samll movie review data
df = pd.read_csv('movie_data_small.csv', encoding='utf-8')

# Set up all data as training data set
X_train = df['review'].values # Contains list of reviews in text
y_train = df['sentiment'].values # Contains review class (0: negative review, 1: positive review)

# Vectorize the movie review data (use words as features)
vect = HashingVectorizer(decode_error='ignore',
                         n_features=2**21, # Specify the number of features to use.  Large value is good to prevent collision of hashed values
                         preprocessor=None,
                         tokenizer=tokenizer)
X_hashed_vecotr_train = vect.transform(X_train) # Contains hashed features (word) for each review

# Use stochastic gradient descent classifier as the model (suitable for incremental learning)
clf = SGDClassifier(loss='log', random_state=1)

# Train the model
clf.fit(X_hashed_vecotr_train, y_train)

# Pickle (save for lator) the stop words
pickle.dump(stop,
            open('pickled_data/stopwords.pkl', 'wb'), # wb: write in binary mode
            protocol=4)

# Pickle (save for lator) the trained model
pickle.dump(clf,
            open('pickled_data/classifier.pkl', 'wb'), # wb: write in binary mode
            protocol=4)
