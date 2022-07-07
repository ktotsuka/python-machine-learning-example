import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Function to compute the output
@tf.function  # decoration to tell Tensorflow to create a static computation graph to improve computational efficiency
def compute_z(a, b, c):
    r1 = tf.subtract(a, b)
    r2 = tf.multiply(2, r1)
    z = tf.add(r2, c)
    return z

# Compute the result with some inputs
tf.print('Scalar Inputs:', compute_z(1, 2, 3))
tf.print('Rank 1 Inputs:', compute_z([1, 2], [2, 3], [3, 4]))
tf.print('Rank 2 Inputs:', compute_z([[1, 2],[3, 4]], [[2, 3],[4, 5]], [[3, 4], [5, 6]]))

# Function to compute the output
# This time, restrict the input so that only rank 1 inputs are allowed
@tf.function(input_signature=(tf.TensorSpec(shape=[None], dtype=tf.int32),
                              tf.TensorSpec(shape=[None], dtype=tf.int32),
                              tf.TensorSpec(shape=[None], dtype=tf.int32),))
def compute_z(a, b, c):
    r1 = tf.subtract(a, b)
    r2 = tf.multiply(2, r1)
    z = tf.add(r2, c)
    return z

# tf.print('Scalar Inputs:', compute_z(1, 2, 3)) # results in error
tf.print('Rank 1 Inputs:', compute_z([1, 2], [2, 3], [3, 4]))
# tf.print('Rank 2 Inputs:', compute_z([[1, 2],[3, 4]], [[2, 3],[4, 5]], [[3, 4], [5, 6]])) # results in error

