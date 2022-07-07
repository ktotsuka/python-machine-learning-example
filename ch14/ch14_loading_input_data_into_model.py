import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

## Version 1 style

# Create a computation graph
g = tf.Graph()

# Add placeholder nodes to the graph
with g.as_default():
    a = tf.compat.v1.placeholder(shape=None, dtype=tf.int32, name='tf_a')
    b = tf.compat.v1.placeholder(shape=None, dtype=tf.int32, name='tf_b')
    c = tf.compat.v1.placeholder(shape=None, dtype=tf.int32, name='tf_c')
    z = 2*(a - b) + c
    
# Compute the result using a sample input
with tf.compat.v1.Session(graph=g) as sess:
    feed_dict = {a:1, b:2, c:3}
    print('Result: z =', sess.run(z, feed_dict=feed_dict))

## Version 2 style

# Function to compute the result
def compute_z(a, b, c):
    r1 = tf.subtract(a, b)
    r2 = tf.multiply(2, r1)
    z = tf.add(r2, c)
    return z

# Compute the result with different inputs
tf.print('Scalar Inputs:', compute_z(1, 2, 3))
tf.print('Rank 1 Inputs:', compute_z([1, 2], [2, 3], [3, 4]))
tf.print('Rank 2 Inputs:', compute_z([[1, 2],[3, 4]], [[2, 3],[4, 5]], [[3, 4], [5, 6]]))