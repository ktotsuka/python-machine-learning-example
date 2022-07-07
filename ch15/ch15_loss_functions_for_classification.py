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

##### Notes #####

# Probability (P): 0 ~ 1
# Logit (L): -inf ~ inf
# P = 1 / (1 + e^-L), this is the sigmoid function
# L = ln(P / (1 - P))

# There are 3 types of loss functions to choose from when using Keras API
# Binary Crossentropy: for binary classification
# Categorical Crossentropy: for multiclass classification and one-hot encoded labels
# Sparse Categorical Crossentropy: for multiclass classification and integer (sparse) labels

##### Binary Crossentropy #####

# Create loss function (one using probability, the other using logits)
bce_probas = tf.keras.losses.BinaryCrossentropy(from_logits=False)
bce_logits = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# Create a sample logit and the corresponding probability
logits = tf.constant([0.8])
probas = tf.keras.activations.sigmoid(logits)

# Print out the calculated loss
tf.print(
    'BCE (w Probas): {:.4f}'.format(
    bce_probas(y_true=[1], y_pred=probas)),
    '(w Logits): {:.4f}'.format(
    bce_logits(y_true=[1], y_pred=logits)))

##### Categorical Crossentropy #####

# Create loss function (one using probability, the other using logits)
cce_probas = tf.keras.losses.CategoricalCrossentropy(
    from_logits=False)
cce_logits = tf.keras.losses.CategoricalCrossentropy(
    from_logits=True)

# Create a sample logit and the corresponding probability
logits = tf.constant([[1.5, 0.8, 2.1]])
probas = tf.keras.activations.softmax(logits) # for multiclass classification, use softmax

# Print out the calculated loss
tf.print(
     'CCE (w Probas): {:.4f}'.format(
    cce_probas(y_true=[[0, 0, 1]], y_pred=probas)),
    '(w Logits): {:.4f}'.format(
    cce_logits(y_true=[[0, 0, 1]], y_pred=logits)))


##### Sparse Categorical Crossentropy #####

# Create loss function (one using probability, the other using logits)
sp_cce_probas = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=False)
sp_cce_logits = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True)

# Print out the calculated loss
tf.print(
    'Sparse CCE (w Probas): {:.4f}'.format(
    sp_cce_probas(y_true=[2], y_pred=probas)),
    '(w Logits): {:.4f}'.format(
    sp_cce_logits(y_true=[2], y_pred=logits)))
