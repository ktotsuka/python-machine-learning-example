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

# Convolusion function for 2D
def conv2d(X, W, p=(0, 0), s=(1, 1)):
    # Rotate the fileter
    W_rot = np.array(W)[::-1,::-1]
    # Pad the input with 0s
    X_orig = np.array(X)
    n1 = X_orig.shape[0] + 2*p[0] # 1st dimention of padded X
    n2 = X_orig.shape[1] + 2*p[1] # 2nd dimention of padded X
    X_padded = np.zeros(shape=(n1, n2))
    X_padded[p[0]:p[0]+X_orig.shape[0],
    p[1]:p[1]+X_orig.shape[1]] = X_orig
    # Compute the result
    result = []
    for i in range(0, int((X_padded.shape[0] - W_rot.shape[0])/s[0])+1): # see p525 for equation
        result.append([])
        for j in range(0, int((X_padded.shape[1] - W_rot.shape[1])/s[1])+1): # see p525 for equation
            X_sub = X_padded[i*s[0]:i*s[0]+W_rot.shape[0],
                             j*s[1]:j*s[1]+W_rot.shape[1]]
            result[-1].append(np.sum(X_sub * W_rot)) # -1: refer to the last element in an arry
    return(np.array(result))

# Define inputs
# X signal
# W: filter or kernel
X = [[1, 3, 2, 4], [5, 6, 1, 3], [1, 2, 0, 2], [3, 4, 3, 2]]
W = [[1, 0, 3], [1, 2, 1], [0, 1, 1]]

# Compute convolution with function in this file
# p: padding (0s on the ends)
# s: stride 
print('Conv2d Implementation:\n',
    conv2d(X, W, p=(1, 1), s=(2, 2)))

# Compute convolution with Scipy function
# mode='same': apply padding so that the output will have the same size as the input
print('SciPy Results:\n',
    scipy.signal.convolve2d(X, W, mode='same'))