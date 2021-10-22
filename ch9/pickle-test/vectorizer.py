from sklearn.feature_extraction.text import HashingVectorizer
import re
import os
import pickle
from tokenizer import *

# Load the pickled stop words (are, is, and, etc.)
stop = pickle.load(open('pickled_data/stopwords.pkl', 'rb'))

# Instanciate a vectorizer
vect = HashingVectorizer(decode_error='ignore',
                         n_features=2**21, # Specify the number of features to use.  Large value is good to prevent collision of hashed values
                         preprocessor=None,
                         tokenizer=tokenizer)
