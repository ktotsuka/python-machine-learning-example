import tensorflow as tf
import numpy as np
import pathlib
import matplotlib.pyplot as plt
import os
import tensorflow_datasets as tfds

# Create some tensors.  One with uniform distribution.  Another with normal distribution.
tf.random.set_seed(1)
t1 = tf.random.uniform(shape=(5, 2), 
                       minval=-1.0,
                       maxval=1.0)
t2 = tf.random.normal(shape=(5, 2), 
                      mean=0.0,
                      stddev=1.0)

# Compute element-wise product
t3 = tf.multiply(t1, t2).numpy()

# Compute mean (along the first axis)
t4 = tf.math.reduce_mean(t1, axis=0)

# Compute matrix multiplication
t5 = tf.linalg.matmul(t1, t2, transpose_b=True) # transpose t2
t6 = tf.linalg.matmul(t1, t2, transpose_a=True) # transpose t1

# Compute L2 norm along the 2nd axis
norm_t1 = tf.norm(t1, ord=2, axis=1).numpy() # ord=2 -> L2 norm
norm_t1_np = np.sqrt(np.sum(np.square(t1), axis=1)) # same, but using numpy