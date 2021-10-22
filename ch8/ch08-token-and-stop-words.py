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

# Tokenize a sentence using simple text split using a white space
simple_tokenizer_output = tokenizer('runners like running and thus they run')
print(simple_tokenizer_output)

# Tokenize a sentence using simple text split using a white space and porter stemmer (convert to root word)
porter_tokenizer_output = tokenizer_porter('runners like running and thus they run')
print(porter_tokenizer_output)

# Remove stop-words (is, are, and, etc.)
nltk.download('stopwords')
stop = stopwords.words('english')
porter_tokenizer_and_stop_words_output = [w for w in tokenizer_porter('a runner likes running and runs a lot')[-10:] if w not in stop]
print(porter_tokenizer_and_stop_words_output)
