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

# Read a color image using tf.io
img_raw = tf.io.read_file('example-image.png')
img = tf.image.decode_image(img_raw)
print('Image shape:', img.shape)
print('Number of channels:', img.shape[2])
print('Image data type:', img.dtype)
print(img[100:102, 100:102, :]) # print out some image data

# Read a color image using imageio
img = imageio.imread('example-image.png')
print('Image shape:', img.shape)
print('Number of channels:', img.shape[2])
print('Image data type:', img.dtype)
print(img[100:102, 100:102, :]) # print out some image data

# Read a grey-scale image using tf.io
img_raw = tf.io.read_file('example-image-gray.png')
img = tf.image.decode_image(img_raw)
tf.print('Rank:', tf.rank(img)) # note that rank is still 3 (the last dimention (color channels) has only 1 entry)
tf.print('Shape:', img.shape)

# Read a grey-scale image using imageio
img = imageio.imread('example-image-gray.png')
tf.print('Rank:', tf.rank(img)) # note that rank is 2.  There is no dimention for the color channel
tf.print('Shape:', img.shape)

# Transform from 2 dimentional data (no color channel) to 3 dimentional datal (with color channel)
img_reshaped = tf.reshape(img, (img.shape[0], img.shape[1], 1))
tf.print('New Shape:', img_reshaped.shape)