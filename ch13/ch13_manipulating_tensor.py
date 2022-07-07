import tensorflow as tf
import numpy as np
import pathlib
import matplotlib.pyplot as plt
import os
import tensorflow_datasets as tfds

# Create a tensor
a = np.array([1, 2, 3], dtype=np.int32)
t_a = tf.convert_to_tensor(a)

# Change the tensor data type
t_a_new = tf.cast(t_a, tf.int64)
print(t_a_new.dtype)

# Transpose a tensor
t = tf.random.uniform(shape=(3, 5))
t_tr = tf.transpose(t)
print(t.shape, ' --> ', t_tr.shape)

# Reshape a tensor
t = tf.zeros((30,))
t_reshape = tf.reshape(t, shape=(5, 6))
print(t_reshape.shape)

# Squeeze (remove dimentions of size 1) a tensor
t = tf.zeros((1, 2, 1, 4, 1))
t_sqz = tf.squeeze(t, axis=(2, 4)) # remove the 2nd and the 4th dimention (counting starts from 0)
print(t.shape, ' --> ', t_sqz.shape)