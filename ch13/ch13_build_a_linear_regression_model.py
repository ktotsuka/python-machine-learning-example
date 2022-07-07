import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
from scipy.special import expit

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

# Function to train a model using gradient descent
def train(model, inputs, outputs, learning_rate):
    with tf.GradientTape() as tape:
        current_loss = loss_fn(model(inputs), outputs)
    dW, db = tape.gradient(current_loss, [model.w, model.b])
    model.w.assign_sub(learning_rate * dW)
    model.b.assign_sub(learning_rate * db)

################################################################################################

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

# Create a Tensorflow dataset using the fake data
ds_train_orig = tf.data.Dataset.from_tensor_slices(
    (tf.cast(X_train_norm, tf.float32),
     tf.cast(y_train, tf.float32)))

# Instantiate and build the linear model
model = MyModel()
model.build(input_shape=(None, 1)) # None: size of the input sample data can be any number.  1: the dimention of the input is 1
model.summary() # This will output the summary of the model

# Set up parameters for training
num_epochs = 200
log_steps = 100
learning_rate = 0.001
batch_size = 1 # train the model one data point at a time
steps_per_epoch = int(np.ceil(len(y_train) / batch_size))
Ws, bs = [], []

# Create a dataset that are shuffled and repeats forever
ds_train = ds_train_orig.shuffle(buffer_size=len(y_train))
ds_train = ds_train.repeat(count=None) # count=None: repeats forever
ds_train = ds_train.batch(1)

# Train the model
for i, batch in enumerate(ds_train):
    if i >= steps_per_epoch * num_epochs:
        break
    Ws.append(model.w.numpy()) # array that contains the value of the parameter "w" for each training iteration
    bs.append(model.b.numpy()) # array that contains the value of the parameter "b" for each training iteration

    bx, by = batch
    loss_val = loss_fn(model(bx), by) # compute the cost based on the current model

    train(model, bx, by, learning_rate=learning_rate)
    if i % log_steps == 0:
        print('Epoch {:4d} Step {:2d} Loss {:6.4f}'.format(
              int(i/steps_per_epoch), i, loss_val))

# Prints out the training result
print('Final Parameters:', model.w.numpy(), model.b.numpy())

# Create dataset for plotting the predicted values
X_test = np.linspace(0, 9, num=100).reshape(-1, 1) # reshape(-1,1) converts to 2D
X_test_norm = (X_test - np.mean(X_train)) / np.std(X_train) # standardize
y_pred = model(tf.cast(X_test_norm, dtype=tf.float32))

# Plot the actual and predected data
fig = plt.figure(figsize=(13, 5)) # figure size in inches
ax = fig.add_subplot(1, 2, 1)
plt.plot(X_train_norm, y_train, 'o', markersize=10) # plot the training dataset
plt.plot(X_test_norm, y_pred, '--', lw=3) # plot the predicted dataset
plt.legend(['Training examples', 'Linear Reg.'], fontsize=15)
ax.set_xlabel('x', size=15)
ax.set_ylabel('y', size=15)
ax.tick_params(axis='both', which='major', labelsize=15)

# Plot the training of the model parameters
ax = fig.add_subplot(1, 2, 2)
plt.plot(Ws, lw=3)
plt.plot(bs, lw=3)
plt.legend(['Weight w', 'Bias unit b'], fontsize=15)
ax.set_xlabel('Iteration', size=15)
ax.set_ylabel('Value', size=15)
ax.tick_params(axis='both', which='major', labelsize=15)

plt.show(block=True)
