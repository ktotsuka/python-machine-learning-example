import tensorflow as tf
import numpy as np
import pathlib
import matplotlib.pyplot as plt
import os
import tensorflow_datasets as tfds

# The model is "y = w*x + b"
class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.w = tf.Variable(0.0, name='weight')
        self.b = tf.Variable(0.0, name='bias')

    def call(self, x):
        return self.w*x + self.b

# Cost function (mean squared error)
def loss_fn(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

######################################################################

# Create fake data for linear regression
X_train = np.arange(10) # create 1D array of 0 ~ 9
y_train = np.array([1.0, 1.3, 3.1,
                    2.0, 5.0, 6.3,
                    6.6, 7.4, 8.0,
                    9.0])

# Plot the fake data
plt.plot(X_train, y_train, 'o', markersize=10)
plt.xlabel('x')
plt.ylabel('y')
plt.show(block=True)

# Standardize the input
X_train_norm = (X_train - np.mean(X_train))/np.std(X_train)

# Instantiate the model
model = MyModel()

# Configures the model for training
model.compile(optimizer='sgd', # use stochastic gradient desent
              loss=loss_fn) # cost function to use

# Set up parameters for training
num_epochs = 200 # go through the training data 200 times for training
batch_size = 1 # train the model one data point at a time

# Train the model
model.fit(X_train_norm, y_train, 
          epochs=num_epochs, batch_size=batch_size,
          verbose=1)

# Print out the final model parameters
print(model.w.numpy(), model.b.numpy())

# Create dataset for plotting the predicted values
X_test = np.linspace(0, 9, num=100) # reshape(-1,1) converts to 2D
X_test_norm = (X_test - np.mean(X_train)) / np.std(X_train)
y_pred = model(tf.cast(X_test_norm, dtype=tf.float32))

# Plot the actual and predected data
fig = plt.figure(figsize=(13, 5)) # figure size of 13" x 5"
ax = fig.add_subplot(1, 2, 1)
plt.plot(X_train_norm, y_train, 'o', markersize=10)
plt.plot(X_test_norm, y_pred, '--', lw=3)
plt.legend(['Training Samples', 'Linear Regression'], fontsize=15)
plt.show(block=True)
