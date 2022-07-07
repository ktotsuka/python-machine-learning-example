from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import sys
import gzip
import shutil
import os
import struct
import numpy as np
import matplotlib.pyplot as plt
from neuralnet import NeuralNetMLP

# Load the npz image data (images of hand-written digit)
# X: pixel value transformed to have gray scale values from -1 (white) to 1 (black)
# y: label (0, 1, .., 9)
mnist = np.load('mnist_scaled.npz')
X_train, y_train, X_test, y_test = [mnist[f] for f in ['X_train', 'y_train', 'X_test', 'y_test']]
del mnist

# Train the neural network MLP model (page 388)
# Use neural network with 3 layers (input, hidden and output layers)
# Input layer has m+1 units (m = 784 = number of pixel in a image)
# Hidden layer has d+1 units (d = 100 (this is a hyperparameter))
# Output layer has t units (t = 10 = digit 0 ~ 9)
#
# The validation data set is not needed to train the model, but it is used to track the accuracy of the model for each epoch of training
# The hidden layer is used to generate a better features than the original features. 
# Using the activation function (sigmoid), the neural network can handle nonlinear functions.
nn = NeuralNetMLP(n_hidden=100, # number of hidden units -1 (=d)
                  l2=0.01, # L2 regularization parameter to decrease the degree of overfitting
                  epochs=200, # number of passes over the training data set
                  eta=0.0005, # the learning rate.  Bigger number helps to escape from local minimums, but increase the chance of overshooting the global minimum.
                  minibatch_size=100, # The number of training examples to use for updating the weights using SGD (stochastic gradient descent)
                                      # The gradient is computed for each mini-batch separately instead of the entire training data for faster learning
                  shuffle=True, # this is for shuffling the training data set prior to every epoch to prevent the algorithm getting stuck in circles
                  seed=1)
nn.fit(X_subtrain=X_train[:55000], # input data for sub-training
       y_subtrain=y_train[:55000], # label for sub-training 
       X_valid=X_train[55000:], # input data for validation
       y_valid=y_train[55000:]) # label for validation

# Plot the cost (error) vs. epoc
plt.figure(1)
plt.plot(range(nn.epochs), nn.eval_['cost'])
plt.ylabel('Cost')
plt.xlabel('Epochs')

# Plot the accuracy vs epochs for sub-training and validation data set
plt.figure(2)
plt.plot(range(nn.epochs), nn.eval_['subtrain_acc'], 
         label='Sub-training')
plt.plot(range(nn.epochs), nn.eval_['valid_acc'], 
         label='Validation', linestyle='--')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(loc='lower right')

# Evaluate the model with the testing data set
y_test_pred = nn.predict(X_test)
acc = (np.sum(y_test == y_test_pred)
       .astype(np.float) / X_test.shape[0])
print('Test accuracy: %.2f%%' % (acc * 100))

# Plot 25 misclassified images
miscl_img = X_test[y_test != y_test_pred][:25]
correct_lab = y_test[y_test != y_test_pred][:25]
miscl_lab = y_test_pred[y_test != y_test_pred][:25]
fig, ax = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True)
ax = ax.flatten() # flatten the subplot from 5x5 to 25x1 to make it easier to iterate them
for i in range(25):
    img = miscl_img[i].reshape(28, 28)
    ax[i].imshow(img, cmap='Greys', interpolation='nearest')
    ax[i].set_title('%d) t: %d p: %d' % (i+1, correct_lab[i], miscl_lab[i])) # t: true label, p: predicted label

ax[0].set_xticks([]) # remove x ticks
ax[0].set_yticks([]) # remove y ticks
plt.tight_layout()

plt.show()