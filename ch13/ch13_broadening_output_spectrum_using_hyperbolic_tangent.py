import tensorflow as tf
import numpy as np
import pathlib
import matplotlib.pyplot as plt
import os
import tensorflow_datasets as tfds
from scipy.special import expit

# Sigmoidal logistic function (an activation function that can be used for a hidden layer of NN)
# The output ranges from 0 to 1
def logistic(z):
    return 1.0 / (1.0 + np.exp(-z))

# Hyperbolic tangent function (an activation function that can be used for a hidden layer of NN)
# This function is similar to sigmoidal logistic function, but the output ranges from -1 to 1 instead of 0 to 1
def tanh(z):
    e_p = np.exp(z)
    e_m = np.exp(-z)
    return (e_p - e_m) / (e_p + e_m)

# Create the input for plotting activation functions
z = np.arange(-5, 5, 0.005)

# Compute the output of the activation functions
log_act = logistic(z)
tanh_act = tanh(z)

# Plot the two activation functions
plt.ylim([-1.5, 1.5])
plt.xlabel('Net input $z$')
plt.ylabel('Activation $\phi(z)$')
plt.axhline(1, color='black', linestyle=':') # create a horizontal line at y = 1
plt.axhline(0.5, color='black', linestyle=':')
plt.axhline(0, color='black', linestyle=':')
plt.axhline(-0.5, color='black', linestyle=':')
plt.axhline(-1, color='black', linestyle=':')
plt.plot(z, tanh_act,
    linewidth=3, linestyle='--',
    label='Tanh')
plt.plot(z, log_act,
    linewidth=3,
    label='Logistic')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show(block=True)

# More practical way to compute the hyperbolic tangent function
np.tanh(z)
tf.keras.activations.tanh(z)

# More practical way to compute the sigmoidal logistic function
expit(z)
tf.keras.activations.sigmoid(z)