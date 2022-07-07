import tensorflow as tf
import numpy as np
import pathlib
import matplotlib.pyplot as plt
import os
import tensorflow_datasets as tfds

# Create a fake dataset
a = [1.2, 3.4, 7.5, 4.1, 5.0, 1.0]

# Convert into a Tensorflow dataset
ds = tf.data.Dataset.from_tensor_slices(a)

# Print items in the dataset
for item in ds:
    print(item)

# Set up batches for the dataset
ds_batch = ds.batch(3) # batch_size: 3

# Iterate through the batches and print them
for i, elem in enumerate(ds_batch, 1): # start index for i: 1
    print('batch {}:'.format(i), elem.numpy()) # "i" is substituted for "{}"