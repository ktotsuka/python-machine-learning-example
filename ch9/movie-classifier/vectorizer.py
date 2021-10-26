from sklearn.feature_extraction.text import HashingVectorizer
import re
import os
import pickle

def tokenizer(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',
                           text.lower())
    text = re.sub('[\W]+', ' ', text.lower()) \
                   + ' '.join(emoticons).replace('-', '')
    tokenized = [w for w in text.split() if w not in stop]
    return tokenized

# 
cur_dir = os.path.dirname(__file__)
stop = pickle.load(open(
                os.path.join(cur_dir, 
                'pkl_objects', 
                'stopwords.pkl'), 'rb'))


# Instanciate a vectorizer
vect = HashingVectorizer(decode_error='ignore',
                         n_features=2**21, # Specify the number of features to use.  Large value is good to prevent collision of hashed values
                         preprocessor=None,
                         tokenizer=tokenizer)
