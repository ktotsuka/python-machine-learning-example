import pickle
import re
import os
from vectorizer import vect
import numpy as np

# Load the pickled classifier
clf = pickle.load(open('pickled_data/classifier.pkl', 'rb')) # rb: read in binary mode

# Vectorize an example sentence
example = ['I love this movie']
X = vect.transform(example)

# Predict the output using the reloaded classifier
label = {0: 'negative', 1: 'positive'}
print('Prediction: %s\nProbability: %.2f%%' %
      (label[clf.predict(X)[0]],
       np.max(clf.predict_proba(X)) * 100)) # use np.max to get the highest probability, which is the one predicted
