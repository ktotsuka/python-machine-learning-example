import tensorflow as tf
import numpy as np
import pathlib
import matplotlib.pyplot as plt
import os
import tensorflow_datasets as tfds

# Seed the random generator so that shuffling of data is consistent for each execution of this program
tf.random.set_seed(1)

# Load the Iris data (data used for flower classification) and print the information about the data
iris, iris_info = tfds.load('iris', with_info=True)
print(iris_info)

# Print the original data
ds_orig = iris['train']
for item in ds_orig:
    print(item)
print('The original dataset has', len(list(ds_orig)), 'items')
print()

# The Iris data contains only the 'train' dataset with 150 items.  We will first shuffle the dataset, then we will split it into a training dataset 
# with 100 items and a testing dataset with 50 items
ds_orig = ds_orig.shuffle(buffer_size=len(list(ds_orig)), # shuffle the whole dataset
                          reshuffle_each_iteration=False) # False: shuffle only once and each time this dataset is accessed it will not be reshuffled again
ds_train_orig = ds_orig.take(100) # take the first 100 items
ds_test = ds_orig.skip(100) # skip the first 100 items and take the rest

# Transform the training and testing datasets into the form that is acceptable for the fit() function (dictionary -> tuple)
ds_train_orig = ds_train_orig.map(
    lambda x: (x['features'], x['label']))
ds_test = ds_test.map(
    lambda x: (x['features'], x['label']))

# Create the model with two layers (one hidden, one output)
# Layers are "Dense" layer, which is also known as "Fully Connected (FC)" layer.  It is a linear layer that can be represented by f(w*x + b),
# where x is the input features, w is the weight matrix, b is the bias vector, f is the activation function
iris_model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, # dimentionality of output space
                          activation='sigmoid', # Use sigmoid function as the activation function
                          name='fc1', # hidden layer name
                          input_shape=(4,)), # There are 4 features in the Iris dataset
    tf.keras.layers.Dense(3, # dimentionality of output space (there are 3 flower labels for the Iris dataset)
                          name='fc2', # output layer name
                          activation='softmax')]) # Use softmax function as the activation function.  Softmax is good for multi-class classification

# Print out the summary of the model
# fc1 has 80 parameters (4x16 for the weights and 1x16 for the bias)
# fc2 has 51 parameters (16x3 for the weights and 1x3 for the bias)           
iris_model.summary()

# Compile the model
iris_model.compile(optimizer='adam', # name of the optimizer used
                   loss='sparse_categorical_crossentropy', # name of the cost function used
                   metrics=['accuracy']) # list of metrics to keep track of so that you can plot the training progress later

# Set up the training parameters
num_epochs = 100 # number of times to go over the training dataset for training
training_size = len(list(ds_train_orig))
batch_size = 2 # size of items to use for each model parameter adjustment
steps_per_epoch = np.ceil(training_size / batch_size)

# Prepare the training dataset
ds_train = ds_train_orig.shuffle(buffer_size=training_size)
ds_train = ds_train.repeat()
ds_train = ds_train.batch(batch_size=batch_size)
ds_train = ds_train.prefetch(buffer_size=1000) # during training, allow prefetching next element while processing the current one. 
                                               # it improves latency and throughput at the cost of using additional memory

# Train the model
history = iris_model.fit(ds_train, epochs=num_epochs,
                         steps_per_epoch=steps_per_epoch, 
                         verbose=0)

# Save training history data into a variable
hist = history.history

# Plot the fit
fig = plt.figure(figsize=(12, 5)) # figure size in inches
ax = fig.add_subplot(1, 2, 1) # for plotting the loss vs epoch
ax.plot(hist['loss'], lw=3)
ax.set_title('Training loss', size=15)
ax.set_xlabel('Epoch', size=15)
ax.tick_params(axis='both', which='major', labelsize=15)

ax = fig.add_subplot(1, 2, 2) # for plotting the accuracy vs epoch
ax.plot(hist['accuracy'], lw=3)
ax.set_title('Training accuracy', size=15)
ax.set_xlabel('Epoch', size=15)
ax.tick_params(axis='both', which='major', labelsize=15)
plt.tight_layout()

plt.show(block=True)

# Evaluating the trained model on the test dataset
results = iris_model.evaluate(ds_test.batch(50), verbose=0) # 50 -> all of the testing dataset
print('Test loss: {:.4f}   Test Acc.: {:.4f}'.format(*results))

# Save the trained model
iris_model.save('iris-classifier.h5',
                overwrite=True,
                include_optimizer=True,
                save_format='h5') # save as HDF5 format