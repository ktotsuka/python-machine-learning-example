import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
from tensorflow_datasets.core.utils.gcs_utils import gcs_dataset_info_files

# Function for preprocessing image data
def preprocess(sample, size=(64, 64), mode='train'):
    # Get the image and the label (Male or Female)
    image = sample['image']
    label = sample['attributes']['Male']
    # Apply transformations: random crop -> resize to specified size -> random flip
    if mode == 'train':
        image_cropped = tf.image.random_crop(
            image, size=(178, 178, 3)) # crop size
        image_resized = tf.image.resize(
            image_cropped, size=size)
        image_flip = tf.image.random_flip_left_right(
            image_resized)
        # Image data: original -> uint8 (0 ~ 255), formatted -> float (0 ~ 1)
        # Label data: original -> int64, formatted -> int32
        return (image_flip/255.0, tf.cast(label, tf.int32))    
    else:
        image_cropped = tf.image.crop_to_bounding_box(
            image, offset_height=20, offset_width=0,
            target_height=178, target_width=178)
        image_resized = tf.image.resize(
            image_cropped, size=size)
        return (image_resized/255.0, tf.cast(label, tf.int32))

# Seed random generator for consistent outcome
tf.random.set_seed(1)

##### Loading the CelebA dataset #####

# Create a builder for CelebA dataset (images of celebrities' faces)
gcs_base_dir = "gs://celeb_a_dataset/"
celeba_bldr = tfds.builder('celeb_a', data_dir=gcs_base_dir, version='2.0.0')

# Download the data, prepare it, and write it to disk
celeba_bldr.download_and_prepare()
celeba = celeba_bldr.as_dataset(shuffle_files=False)
print(celeba.keys())

# Get the training, validation, and test dataset
celeba_train = celeba['train']
celeba_valid = celeba['validation']
celeba_test = celeba['test']

# To speed up the process, only use part of the training and validation dataset
TRAINING_DATA_SIZE = 16000
VALIDATION_DATA_SIZE = 1000
celeba_train = celeba_train.take(TRAINING_DATA_SIZE)
celeba_valid = celeba_valid.take(VALIDATION_DATA_SIZE)

##### Preprocess training and validation datasets #####

# Set up training parameters
BATCH_SIZE = 32
BUFFER_SIZE = 1000
IMAGE_SIZE = (64, 64)
steps_per_epoch = np.ceil(TRAINING_DATA_SIZE/BATCH_SIZE)
print('Steps per epoch: ', steps_per_epoch)

# Preprocess training dataset
ds_train = celeba_train.map(
    lambda x: preprocess(x, size=IMAGE_SIZE, mode='train'))
ds_train = ds_train.shuffle(buffer_size=BUFFER_SIZE).repeat()
ds_train = ds_train.batch(BATCH_SIZE)

# Preprocess validation dataset
ds_valid = celeba_valid.map(
    lambda x: preprocess(x, size=IMAGE_SIZE, mode='eval'))
ds_valid = ds_valid.batch(BATCH_SIZE)

##### Create a model #####

# 1st layer (2D convolutional layer)
#   input: 64 x 64 x 3 (the three color channels are added to create a feature map, so the color channel dimention goes away)
#   kernel window: 3 x 3
#   output: 64 x 64 x 32
#
# 2nd layer (max pooling layer)
#   input: 64 x 64 x 32
#   pooling size: 2 x 2
#   ouput: 32 x 32 x 32
#
# 3rd layer (dropout layer)
#   input: 32 x 32 x 32
#   ouput: 32 x 32 x 32
#
# 4th layer (2D convolutional layer)
#   input: 32 x 32 x 32
#   kernel window: 3 x 3
#   output: 32 x 32 x 64
#
# 5th layer (max pooling layer)
#   input: 32 x 32 x 64
#   pooling size: 2 x 2
#   ouput: 16 x 16 x 64
#
# 6th layer (dropout layer)
#   input: 16 x 16 x 64
#   ouput: 16 x 16 x 64
#
# 7th layer (2D convolutional layer)
#   input: 16 x 16 x 64
#   kernel window: 3 x 3
#   output: 16 x 16 x 128
#
# 8th layer (max pooling layer)
#   input: 16 x 16 x 128
#   pooling size: 2 x 2
#   ouput: 8 x 8 x 128
#
# 9th layer (2D convolutional layer)
#   input: 8 x 8 x 128
#   kernel window: 3 x 3
#   output: 8 x 8 x 256
#
model = tf.keras.Sequential()

model.add(tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Dropout(rate=0.5))

model.add(tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Dropout(rate=0.5))

model.add(tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu'))
print('Output shape after convolutional layers', model.compute_output_shape(input_shape=(None, 64, 64, 3)))

# 10th layer (global average pooling layer)
#   input: 8 x 8 x 256
#   output: 256
#
model.add(tf.keras.layers.GlobalAveragePooling2D())
print('Output shape after global average pooling layers', model.compute_output_shape(input_shape=(None, 64, 64, 3)))

# 11th layer (fully connected layer)
#   input: 256
#   output: 1 (since no activation function was used, the output will be logits (not probability))
# 
model.add(tf.keras.layers.Dense(1, activation=None))

# Build the model
model.build(input_shape=(None, 64, 64, 3))
model.summary()

##### Compile the model #####

# loss: Since the output will be binary classification, use "BinaryCrossentropy"
# Since we did not use activation function for the last layer, "from_logits" is set to True
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

##### Train the model #####
NUM_EPOCH = 20
history = model.fit(ds_train, validation_data=ds_valid, # this step takes a long time (4 hours?)
                    epochs=NUM_EPOCH, steps_per_epoch=steps_per_epoch)

##### Plot the fit #####

hist = history.history
x_arr = np.arange(NUM_EPOCH) + 1

# Create a figure
fig = plt.figure(figsize=(12, 4)) # figure size in inches

# Plot the loss
ax = fig.add_subplot(1, 2, 1)
ax.plot(x_arr, hist['loss'], '-o', label='Train loss')
ax.plot(x_arr, hist['val_loss'], '--<', label='Validation loss')
ax.legend(fontsize=15)
ax.set_xlabel('Epoch', size=15)
ax.set_ylabel('Loss', size=15)

# Plot the accuracy
ax = fig.add_subplot(1, 2, 2)
ax.plot(x_arr, hist['accuracy'], '-o', label='Train acc.')
ax.plot(x_arr, hist['val_accuracy'], '--<', label='Validation acc.')
ax.legend(fontsize=15)
ax.set_xlabel('Epoch', size=15)
ax.set_ylabel('Accuracy', size=15)

plt.show(block=True)

##### Train the model more #####

# Train for 10 more epochs
EXTRA_EPOCH = 10
history = model.fit(ds_train, validation_data=ds_valid, # this step takes a long time (2 hours)
                    epochs=(NUM_EPOCH + EXTRA_EPOCH), initial_epoch=NUM_EPOCH,
                    steps_per_epoch=steps_per_epoch)

##### Plot the fit (again) #####

hist2 = history.history
x_arr = np.arange(NUM_EPOCH + EXTRA_EPOCH)

# Create a figure
fig = plt.figure(figsize=(12, 4))

# Plot the loss
ax = fig.add_subplot(1, 2, 1)
ax.plot(x_arr, hist['loss']+hist2['loss'], 
        '-o', label='Train Loss')
ax.plot(x_arr, hist['val_loss']+hist2['val_loss'],
        '--<', label='Validation Loss')
ax.legend(fontsize=15)

# Plot the accuracy
ax = fig.add_subplot(1, 2, 2)
ax.plot(x_arr, hist['accuracy']+hist2['accuracy'], 
        '-o', label='Train Acc.')
ax.plot(x_arr, hist['val_accuracy']+hist2['val_accuracy'], 
        '--<', label='Validation Acc.')
ax.legend(fontsize=15)

plt.show(block=True)

##### Evaluate the test dataset #####

# Preprocess the test dataset 
ds_test = celeba_test.map(
    lambda x:preprocess(x, size=IMAGE_SIZE, mode='eval')).batch(32)

# Evaluate the test dataset
results = model.evaluate(ds_test, verbose=0) # this step takes a long time (hours)
print('Test Acc: {:.2f}%'.format(results[1]*100))

##### Predict the gender from samples from the testing dataset #####

# Get 10 sample images from the test dataset
# Need to unbatch first because we set the batch size to be 32 previously.
# If no unbatch, we would get 10 batches with each batch contains 32 sample images
ds = ds_test.unbatch().take(10) 

# Predict the gender
pred_logits = model.predict(ds.batch(10)) # this step takes a long time (2 hours)
probas = tf.sigmoid(pred_logits) # convert logits to probability
probas = probas.numpy().flatten()*100

##### Display the prediction results #####

fig = plt.figure(figsize=(15, 7)) # figure size in inches
for j,example in enumerate(ds): # this step takes a long time (2 minutes)
    ax = fig.add_subplot(2, 5, j+1)
    ax.set_xticks([]); ax.set_yticks([])
    ax.imshow(example[0])
    if example[1].numpy() == 1:
        label='Male'
    else:
        label = 'Female'
    ax.text(
        0.5, -0.15, 
        'GT: {:s}\nPr(Male)={:.0f}%'.format(label, probas[j]), 
        size=16, 
        horizontalalignment='center',
        verticalalignment='center', 
        transform=ax.transAxes)
    
plt.show(block=True)
