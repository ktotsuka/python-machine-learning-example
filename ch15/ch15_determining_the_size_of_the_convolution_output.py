import tensorflow as tf
import numpy as np
import scipy.signal
import imageio
from tensorflow import keras
import tensorflow_datasets as tfds
import pandas as pd
import matplotlib.pyplot as plt
import os
from distutils.version import LooseVersion as Version

# Convolusion function for 1D
def conv1d(x, w, p=0, s=1):
    w_rot = np.array(w[::-1])
    x_padded = np.array(x)
    if p > 0:
        zero_pad = np.zeros(shape=p)
        x_padded = np.concatenate(
            [zero_pad, x_padded, zero_pad])
    result = []
    for i in range(0, int((len(x_padded) - len(w_rot)) / s) + 1): # see p525 for equation
        result.append(np.sum(
            x_padded[i*s:i*s+w_rot.shape[0]] * w_rot))
    return np.array(result)

# Define inputs
# x: signal
# w: filter or kernel
x = [1, 3, 2, 4, 5, 6, 1, 3]
w = [1, 0, 3, 1, 2]

# Compute convolution with function in this file
# p: padding (0s on the ends)
# s: stride 
print('Conv1d Implementation:',
      conv1d(x, w, p=2, s=2))

# Compute convolution with Numpy function
# mode='same': apply padding so that the output will have the same size as the input
print('Numpy Results:',
      np.convolve(x, w, mode='same')) 

