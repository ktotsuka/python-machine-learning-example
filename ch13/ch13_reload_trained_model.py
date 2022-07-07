import tensorflow as tf
import numpy as np
import pathlib
import matplotlib.pyplot as plt
import os
import tensorflow_datasets as tfds

# Seed the random generator so that shuffling of data is consistent for each execution of this program
tf.random.set_seed(1)

# Load the Iris data (data used for flower classification) and print the information about the data
iris, iris_info = tfds.load('iris', with_info=True)

# The Iris data contains only the 'train' dataset with 150 items.  We will first shuffle the dataset, then we will split it into a training dataset 
# with 100 items and a testing dataset with 50 items
ds_orig = iris['train']
ds_orig = ds_orig.shuffle(buffer_size=len(list(ds_orig)), # shuffle the whole dataset
                          reshuffle_each_iteration=False) # False: shuffle only once and each time this dataset is accessed it will not be reshuffled again
ds_train_orig = ds_orig.take(100) # take the first 100 items
ds_test = ds_orig.skip(100) # skip the first 100 items and take the rest

# Transform the training and testing datasets into the form that is acceptable for the fit() function (dictionary -> tuple)
ds_train_orig = ds_train_orig.map(
    lambda x: (x['features'], x['label']))
ds_test = ds_test.map(
    lambda x: (x['features'], x['label']))

# Load the saved model (Iris classifier)
iris_model_new = tf.keras.models.load_model('iris-classifier.h5')

# Print the model summry
iris_model_new.summary()

# Evaluate the loaded model
results = iris_model_new.evaluate(ds_test.batch(50), verbose=0)
print('Test loss: {:.4f}   Test Acc.: {:.4f}'.format(*results))

# Print the model information in json format
print(iris_model_new.to_json())