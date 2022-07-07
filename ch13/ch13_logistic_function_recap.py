import tensorflow as tf
import numpy as np
import pathlib
import matplotlib.pyplot as plt
import os
import tensorflow_datasets as tfds

# Compute z = w0*x0 + w1*x1 + ... + wm*xm
def net_input(X, w):
    return np.dot(X, w)

# The sigmoid function
def logistic(z):
    return 1.0 / (1.0 + np.exp(-z))

def logistic_activation(X, w):
    z = net_input(X, w)
    return logistic(z)

# The softmax function
def softmax(z):
    return np.exp(z) / np.sum(np.exp(z))

######### Case where output size is 1 (label = 0 or 1) with sigmoid logistic activation #####################

# Set up inputs
# z = w0*x0 + w1*x1 + ... + wm*xm
X = np.array([1, 1.4, 2.5]) # input.  The first value (x0) is for the bias term and must be 1.  m=2 in this case
w = np.array([0.4, 0.3, 0.5]) # weight.  The first term (w0) is the bias

# Compute teh probability of output, y, being 1 (instead of 0)
print('P(y=1|x) = %.3f' % logistic_activation(X, w)) 

######### Case where output size is 3 (label = 0, 1, or 2 -> use one-hot encoding) with sigmoid logistic activation ###################

# W: weight matrix (num_outputs, m+1)
# Note that the first column are the bias units
W = np.array([[1.1, 1.2, 0.8, 0.4], # m = 3
              [0.2, 0.4, 1.0, 0.2],
              [0.6, 1.5, 1.2, 0.7]])

# A : input (m + 1, n_samples)
# The first value (x0) is for the bias term and must be 1
A = np.array([[1, 0.1, 0.4, 0.6]]) # m = 3, number of sample is 1
Z = np.dot(W, A[0])
y_probas = logistic(Z) # this is not the probability of output, y0, y1, y2 being 1 because they don't sum up to 1.  
                       # But you can use it to predict the most likely output label
print('Net Input: \n', Z)
print('Output Units:\n', y_probas) 
print('Sum of probability for each output label for sigmoid logistic function:', np.sum(y_probas))

# Predict the most likely output label
y_class = np.argmax(Z, axis=0)
print('Predicted class label: %d' % y_class) 

######### Case where output size is 3 (label = 0, 1, or 2 -> use one-hot encoding) with softmax activation ###################
# Compute probability for the output using softmax activation function
y_probas = softmax(Z)
print('Probabilities using softmax:\n', y_probas)

# Show that the sum of the probability for each label is 1
print('sum of probability for each output label using softmax: ', np.sum(y_probas))

# Same thing but using Tensorflow build-in functions
Z_tensor = tf.expand_dims(Z, axis=0)
y_probas_tf = tf.keras.activations.softmax(Z_tensor)
print('Probabilities using softmax from tf:\n', y_probas_tf)
