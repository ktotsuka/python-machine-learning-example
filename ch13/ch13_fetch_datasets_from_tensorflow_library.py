import tensorflow as tf
import numpy as np
import pathlib
import matplotlib.pyplot as plt
import os
import tensorflow_datasets as tfds

# Check the number of available dataset from the Tensorflow library and print out the name of first five
print(len(tfds.list_builders()))
print(tfds.list_builders()[:5])

# Create a builder for CelebA dataset (images of celebrities' faces)
gcs_base_dir = "gs://celeb_a_dataset/"
celeba_bldr = tfds.builder('celeb_a', data_dir=gcs_base_dir, version='2.0.0')

# Print out various info about the CelebA dataset
print(celeba_bldr.info.features.keys()) # features consists of "attributes", "image", "landmarks"
print('\n', 30*"=", '\n')
print(celeba_bldr.info.features['attributes'].keys()) # boolean features of the face
print('\n', 30*"=", '\n')
print(celeba_bldr.info.features['image']) # information about the image size and data type
print('\n', 30*"=", '\n')
print(celeba_bldr.info.features['landmarks'].keys()) # x-y coordinate of facial landmarks
print('\n', 30*"=", '\n')
print(celeba_bldr.info.features) # detailed information for attributes, image, and landmarks
print('\n', 30*"=", '\n')
print(celeba_bldr.info.citation) # information about where the images came from

# Download the data, prepare it, and write it to disk
celeba_bldr.download_and_prepare()

# Load data from disk as Tensorflow datasets
datasets = celeba_bldr.as_dataset(shuffle_files=False)
datasets.keys() # it has "test", "train", and "validation" sets

# Get training dataset
ds_train = datasets['train']
assert isinstance(ds_train, tf.data.Dataset)

# Transform the training dataset so that it consists of the images and the male/female labels
ds_train = ds_train.map(lambda item: 
     (item['image'], tf.cast(item['attributes']['Male'], tf.int32)))

# Grab the first 18 samples from the training set 
ds_train = ds_train.batch(18) # transform the training dataset into batches of 18 samples
images, labels = next(iter(ds_train)) # Grab the first batch, which contains 18 samples.  This takes several minutes
print(images.shape, labels) # image shape is 218 x 178 x 3 (RGB), label is 0=female, 1=male

# Display the sample images
fig = plt.figure(figsize=(12, 8)) # figure size in inches
for i,(image,label) in enumerate(zip(images, labels)):
    ax = fig.add_subplot(3, 6, i+1) # 3 x 6 = 18 images
    ax.set_xticks([]); ax.set_yticks([]) # clear x-y axis ticks
    ax.imshow(image)
    ax.set_title('{}'.format(label), size=15) # set the label (0=female, 1=male) as the title
plt.show(block=True)

# Load MNIST dataset (images of single digit numbers)
# tfds.load = tfds.builder + builder.download_and_prepare + builder.as_dataset
mnist, mnist_info = tfds.load('mnist', with_info=True,
                              shuffle_files=False)
print(mnist_info)
print(mnist.keys()) # the keys are "test" and "train" for datasets

# Get the training dataset
ds_train = mnist['train']
assert isinstance(ds_train, tf.data.Dataset)

# Transform the training dataset so that it consists of the images and the labels (single digit that the image represents)
ds_train = ds_train.map(lambda item: 
     (item['image'], item['label']))

# Grab the first 10 samples from the training set 
ds_train = ds_train.batch(10) # transform the training dataset into batches of 10 samples
batch = next(iter(ds_train)) # Grab the first batch, which contains 10 samples
print(batch[0].shape, batch[1]) # image size is 28x28x1 (single color), label is the single digit from 0 to 9

# Display the sample images
fig = plt.figure(figsize=(15, 6)) # figure size in inches
for i,(image,label) in enumerate(zip(batch[0], batch[1])):
    ax = fig.add_subplot(2, 5, i+1) # 2 x 5 = 10 images
    ax.set_xticks([]); ax.set_yticks([]) # clear x-y axis ticks
    ax.imshow(image[:, :, 0], cmap='gray_r')
    ax.set_title('{}'.format(label), size=15) # set the label (0=female, 1=male) as the title
plt.show(block=True)
