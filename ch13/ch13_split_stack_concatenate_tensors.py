import tensorflow as tf
import numpy as np
import pathlib
import matplotlib.pyplot as plt
import os
import tensorflow_datasets as tfds

# Create a fake uniform data of size 6
tf.random.set_seed(1)
t = tf.random.uniform((6,)) # default is [0 to 1)

# Split the data into 3 equal parts
t_splits = tf.split(t, 3)

# Create a fake data of size 5
tf.random.set_seed(1)
t = tf.random.uniform((5,))

# Split the data into 2 with different sizes (3 and 2)
t_splits = tf.split(t, num_or_size_splits=[3, 2])

# Concatenate two tensors
A = tf.ones((3,))
B = tf.zeros((2,))
C = tf.concat([A, B], axis=0)

# Stack two tensors
A = tf.ones((3,))
B = tf.zeros((3,))
S = tf.stack([A, B], axis=1)
