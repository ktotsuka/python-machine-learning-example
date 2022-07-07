import tensorflow as tf
import numpy as np
import pathlib
import matplotlib.pyplot as plt
import os
import tensorflow_datasets as tfds

# Create a fake dataset for input, x, and output, y
tf.random.set_seed(1)
t_x = tf.random.uniform([4, 3], dtype=tf.float32) # x: 4x3.  The range is [0 ~ 1), which is the default
t_y = tf.range(4) # y: 4x1

# Create a joint dataset for x and y
ds_x = tf.data.Dataset.from_tensor_slices(t_x)
ds_y = tf.data.Dataset.from_tensor_slices(t_y)    
ds_joint = tf.data.Dataset.zip((ds_x, ds_y))

# Alternate way to create the same joint dataset
# ds_joint = tf.data.Dataset.from_tensor_slices((t_x, t_y))

# Print out the joint dataset
for example in ds_joint:
    print('  x: ', example[0].numpy(), 
          '  y: ', example[1].numpy())

# Transform x so that it has [-1 ~ 1)
ds_trans = ds_joint.map(lambda x, y: (x*2-1.0, y))

# Print out the transformed joint dataset
for example in ds_trans:
    print('  x: ', example[0].numpy(), 
          '  y: ', example[1].numpy())