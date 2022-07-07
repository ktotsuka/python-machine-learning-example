import tensorflow as tf
import numpy as np
import pathlib
import matplotlib.pyplot as plt
import os
import tensorflow_datasets as tfds

print('TensorFlow version:', tf.__version__)

# Set up precision for printing
np.set_printoptions(precision=3)

# Creating tensors in TensorFlow
a = np.array([1, 2, 3], dtype=np.int32)
b = [4, 5, 6]
t_a = tf.convert_to_tensor(a)
t_b = tf.convert_to_tensor(b)
print(t_a)
print(t_b)
tf.is_tensor(a) # false
tf.is_tensor(t_a) # true

# Create more tensors
t_ones = tf.ones((2, 3))
t_ones.shape # TensorShape([2, 3])

# Access numpy array portion of the tensor
t_ones.numpy()

# Create an array directly as a tensor
const_tensor = tf.constant([1.2, 5, np.pi], dtype=tf.float32)
print(const_tensor)


