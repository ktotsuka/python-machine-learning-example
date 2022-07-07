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

# Seed random generator to achieve a consistent result
tf.random.set_seed(1)

##### Download the MNIST dataset #####

# Fetch the MNIST hand-written digit dataset
mnist_bldr = tfds.builder('mnist')
mnist_bldr.download_and_prepare()
datasets = mnist_bldr.as_dataset()

# Get the training and the testing dataset
print(datasets.keys()) # the downloaded data contings 'train' and 'test' datasets
mnist_train_orig, mnist_test_orig = datasets['train'], datasets['test']

##### Prepare the dataset for training #####

# Set up the training parameters
BUFFER_SIZE = 10000
BATCH_SIZE = 64
NUM_EPOCHS = 10

# Format the training and the testing dataset
# The original data is 28x28x1.  Image size is 28x28 and the color is black and white.  
# Image data: original -> uint8 (0 ~ 255), formatted -> float (0 ~ 1)
# Label data: original -> int64, formatted -> int32
mnist_train = mnist_train_orig.map(
    lambda item: (tf.cast(item['image'], tf.float32)/255.0, 
                  tf.cast(item['label'], tf.int32)))
print("Original training dataset size: ", len(list(mnist_train)))

mnist_test = mnist_test_orig.map(
    lambda item: (tf.cast(item['image'], tf.float32)/255.0, 
                  tf.cast(item['label'], tf.int32)))

# Split the original training dataset into the training and the validation dataset
# First, shuffle the original dataset, but don't reshuffle for each iteration 
# so that the training and the validation data set will be split in a consistent manner
mnist_train = mnist_train.shuffle(buffer_size=BUFFER_SIZE,
                                  reshuffle_each_iteration=False)
mnist_valid = mnist_train.take(10000).batch(BATCH_SIZE)
mnist_train = mnist_train.skip(10000).batch(BATCH_SIZE)

##### Define the model (see figure on page 542) #####

# Start with Keras sequential model
model = tf.keras.Sequential()

# Add the first 2D convolutional layer
# filters=32: generate 32 output feature maps
# kernel_size=(5, 5): 5x5 kernel
# strides=(1, 1): step size in each direction (width and height)
# padding='same': the output width and height will be the same as the input
# data_format='channels_last': color channel is the last dimention (as opposed to the first dimention)
# activation='relu'): Use the Rectified Linear Unit (ReLU) activation function
model.add(tf.keras.layers.Conv2D(
    filters=32, kernel_size=(5, 5),
    strides=(1, 1), padding='same',
    data_format='channels_last',
    name='conv_1', activation='relu'))

# Add the pooling layer
# MaxPool2D: pick the max value in the pool window (as opposed to average)
# pool_size=(2, 2): pool window size.  The stride will be the same as the pool size by default
model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), name='pool_1'))
    
# Add the second 2D convolutional layer
model.add(tf.keras.layers.Conv2D(
    filters=64, kernel_size=(5, 5),
    strides=(1, 1), padding='same',
    name='conv_2', activation='relu'))

# Add the pooling layer
model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), name='pool_2'))

# Flatten the data
model.add(tf.keras.layers.Flatten())

# Add the first fully connected (dense) layer
# units=1024: number of outputs
model.add(tf.keras.layers.Dense(
    units=1024, name='fc_1', 
    activation='relu'))

# Add the dropout layer for regularization
# rate=0.5: probability that a parameter will be dropped
model.add(tf.keras.layers.Dropout(
    rate=0.5))
    
# Add the second fully connected (dense) layer
# activation='softmax': Use softmax for multiclass label
model.add(tf.keras.layers.Dense(
    units=10, name='fc_2',
    activation='softmax'))

##### Build and compile the model #####

# Build the model
# None: don't specify the batch size at this point
# 28: image height
# 28: image width
# 1: number of color channels
model.build(input_shape=(None, 28, 28, 1))

# Compute the output shape of the moel based on the given input size
print("Output size: ", model.compute_output_shape(input_shape=(16, 28, 28, 1)))

# Print the model summary
model.summary()

# Compile the model
# Adam: some good optimizer
# loss=tf.keras.losses.SparseCategoricalCrossentropy(): Use SparseCategoricalCrossentropy loss function for multiclass classification 
# and integer (sparse) labels
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

##### Train the model #####

history = model.fit(mnist_train, epochs=NUM_EPOCHS, 
                    validation_data=mnist_valid, 
                    shuffle=True)

##### Plot the model fit #####

# Get the training history
hist = history.history

# Get the x-axis (epoch)
x_arr = np.arange(NUM_EPOCHS) + 1

# Create a figure
fig = plt.figure(figsize=(12, 4)) # figure size in inches

# Plot the loss
ax = fig.add_subplot(1, 2, 1)
ax.plot(x_arr, hist['loss'], '-o', label='Train loss')
ax.plot(x_arr, hist['val_loss'], '--<', label='Validation loss')
ax.set_xlabel('Epoch', size=15)
ax.set_ylabel('Loss', size=15)
ax.legend(fontsize=15)

# Plot the accuracy
ax = fig.add_subplot(1, 2, 2)
ax.plot(x_arr, hist['accuracy'], '-o', label='Train acc.')
ax.plot(x_arr, hist['val_accuracy'], '--<', label='Validation acc.')
ax.set_xlabel('Epoch', size=15)
ax.set_ylabel('Accuracy', size=15)
ax.legend(fontsize=15)

# Show the plot
plt.show(block=True)

##### Evaluate the model with test dataset #####

# Evaluate the model with test dataset and display the accuracy
test_results = model.evaluate(mnist_test.batch(20))
print('\nTest Acc. {:.2f}%'.format(test_results[1]*100))

# Predict the output label using the 1st batch (12 samples) of the test dataset
batch_test = next(iter(mnist_test.batch(12)))
preds = model(batch_test[0]) # the predictions contain the probability for each digit
tf.print(preds.shape)
preds = tf.argmax(preds, axis=1) # get the digit with the highest probability
print(preds)

# Display the images from the 1st batch (12 samples) 
fig = plt.figure(figsize=(12, 4)) # figure size in inches
for i in range(12):
    ax = fig.add_subplot(2, 6, i+1)
    ax.set_xticks([]); ax.set_yticks([]) # remove the x and y axis ticks
    img = batch_test[0][i, :, :, 0]
    ax.imshow(img, cmap='gray_r')
    ax.text(0.9, 0.1, '{}'.format(preds[i]), # add the predicted digit at the lower right corner of the image
            size=15, color='blue',
            horizontalalignment='center',
            verticalalignment='center', 
            transform=ax.transAxes)
    
plt.show(block=True)


##### Save the model #####

if not os.path.exists('models'):
    os.mkdir('models')
model.save('models/mnist-cnn.h5')
