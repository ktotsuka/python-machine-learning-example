import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import pandas as pd
import os
import gzip
import shutil
from collections import Counter
from tensorflow.keras.layers import Embedding
from tensorflow.keras import Sequential
from tensorflow.keras.layers import SimpleRNN
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import GRU
from tensorflow.keras.layers import Bidirectional

# Seed the random generator for consistent results
tf.random.set_seed(1)

##### Import the movie review data #####

# Read the movie review data
# df: Dataframe that contains 50000 pairs of 'review (text)', and 'sentiment (negative review:0, positive review:1)'
df = pd.read_csv('../ch08/movie_data.csv', encoding='utf-8') # df.shape = (50000, 2)

# Create a Tensorflow dataset
target = df.pop('sentiment') # target.shape = (50000,), df.shape = (50000, 1)
ds_raw = tf.data.Dataset.from_tensor_slices((df.values, target.values))

# Print out the first 3 samples from the dataset
for ex in ds_raw.take(3):
    review_text = ex[0].numpy()[0]
    tf.print(review_text[:50], ex[1])

##### Split the dataset into training, validation, and test datasets #####

# Split the dataset into training (20000), validation (5000), and test (25000) sets
ds_raw = ds_raw.shuffle(len(target), reshuffle_each_iteration=False)
ds_raw_test = ds_raw.take(25000)
ds_raw_train_valid = ds_raw.skip(25000)
ds_raw_train = ds_raw_train_valid.take(20000)
ds_raw_valid = ds_raw_train_valid.skip(20000)

##### Preprocess the dataset #####

# Get the list of unique words and their counts in the training dataset
# token_counts: contains the dictionary of word (key) and their count (value)
try:
    tokenizer = tfds.features.text.Tokenizer()
except AttributeError:
    tokenizer = tfds.deprecated.text.Tokenizer()
token_counts = Counter()
for example in ds_raw_train:
    review_text = example[0].numpy()[0]
    tokens = tokenizer.tokenize(review_text) # get list of unique words
    token_counts.update(tokens)    
print('Vocab-size:', len(token_counts))

# Define the function for encoding words (tokens) into integers
try:
    encoder = tfds.features.text.TokenTextEncoder(token_counts)
except AttributeError:
    encoder = tfds.deprecated.text.TokenTextEncoder(token_counts)
def encode(text_tensor, label):
    text = text_tensor.numpy()[0]
    encoded_text = encoder.encode(text)
    return encoded_text, label

# Wrap the encode function to enable eager execution
def encode_map_fn(text, label):
    return tf.py_function(encode, inp=[text, label], Tout=(tf.int64, tf.int64))

# Encode words into integers for all datasets
ds_train = ds_raw_train.map(encode_map_fn)
ds_valid = ds_raw_valid.map(encode_map_fn)
ds_test = ds_raw_test.map(encode_map_fn)

# Take a small subset and print information.  Also show how padded batch works
ds_subset = ds_train.take(8)
for example in ds_subset:
    # example[0]: contains encoded text (word1 word2 word3 word 1) -> (20, 31, 43, 20)
    # example[1]: contains the sentiment (0: negative, 1: positive)
    print('Sequence length (number of words):', example[0].shape)
ds_batched = ds_subset.padded_batch(4, padded_shapes=([-1], []))
for batch in ds_batched:
    print('Batch Shape:', batch[0].shape)

# Batch the datasets with the batch size of 32
# The batches are padded with 0 so that the samples have the same dimention within the batch
train_data = ds_train.padded_batch(32, padded_shapes=([-1],[]))
valid_data = ds_valid.padded_batch(32, padded_shapes=([-1],[]))
test_data = ds_test.padded_batch(32, padded_shapes=([-1],[]))

##### Define an RNN model #####

# Prepare parameters for embedding (reduce input feature dimention)
embedding_dim = 20
vocab_size = len(token_counts) + 2 # number of unique words + 2 (one place holder for padding, another place holder for unseen words)

# Build the model
# 1st layer: embedding layer.  
# 2nd layer: LSTM (a type of RNN layer that is good for a long sequence) set up to process data bidirectionally (forward and backward sequence)
# 3rd layer: fully connected layer
# 4th layer: fully connected layer with output dimention of 1 (sentiment)
bi_lstm_model = tf.keras.Sequential([
    tf.keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        name='embed-layer'),    
    tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(64, name='lstm-layer'),
        name='bidir-lstm'), 
    tf.keras.layers.Dense(64, activation='relu'),    
    tf.keras.layers.Dense(1, activation='sigmoid')
])
bi_lstm_model.summary()

##### Compile the model #####

bi_lstm_model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=False), # loss function for binary output
    metrics=['accuracy'])

##### Train the model #####

history = bi_lstm_model.fit( # this step takes a long time (~5 min)
    train_data, 
    validation_data=valid_data, 
    epochs=5)

##### Evaluate the model #####

# Evaluate the model using the test dataset
test_results= bi_lstm_model.evaluate(test_data)
print('Test Acc.: {:.2f}%'.format(test_results[1]*100))