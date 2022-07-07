import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
from tensorflow_datasets.core.utils.gcs_utils import gcs_dataset_info_files

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
celeba_train = celeba_train.take(10)
celeba_valid = celeba_valid.take(10)

##### Image transformation and data augmentation (example transformations) #####

# Take 5 sample images from the training dataset
samples = []
for sample in celeba_train.take(5): # this step takes a long time (30 min?)
    samples.append(sample['image'])

# Create a figure
fig = plt.figure(figsize=(16, 8.5)) # figure size in inches

# Column 1: cropping to a bounding-box
ax = fig.add_subplot(2, 5, 1)
ax.imshow(samples[0])
ax = fig.add_subplot(2, 5, 6)
ax.set_title('Crop to a \nbounding-box', size=15)
img_cropped = tf.image.crop_to_bounding_box(
    samples[0], 50, 20, 128, 128) # y start, x start, height, width
ax.imshow(img_cropped)

# Column 2: flipping (horizontally)
ax = fig.add_subplot(2, 5, 2)
ax.imshow(samples[1])
ax = fig.add_subplot(2, 5, 7)
ax.set_title('Flip (horizontal)', size=15)
img_flipped = tf.image.flip_left_right(samples[1])
ax.imshow(img_flipped)

# Column 3: adjust contrast
ax = fig.add_subplot(2, 5, 3)
ax.imshow(samples[2])
ax = fig.add_subplot(2, 5, 8)
ax.set_title('Adjust constrast', size=15)
img_adj_contrast = tf.image.adjust_contrast(
    samples[2], contrast_factor=2)
ax.imshow(img_adj_contrast)

# Column 4: adjust brightness
ax = fig.add_subplot(2, 5, 4)
ax.imshow(samples[3])
ax = fig.add_subplot(2, 5, 9)
ax.set_title('Adjust brightness', size=15)
img_adj_brightness = tf.image.adjust_brightness(
    samples[3], delta=0.3)
ax.imshow(img_adj_brightness)

# Column 5: cropping from image center then resize
ax = fig.add_subplot(2, 5, 5)
ax.imshow(samples[4])
ax = fig.add_subplot(2, 5, 10)
ax.set_title('Centeral crop\nand resize', size=15)
img_center_crop = tf.image.central_crop(
    samples[4], 0.7)
img_resized = tf.image.resize(
    img_center_crop, size=(218, 178))
ax.imshow(img_resized.numpy().astype('uint8'))

plt.show(block=True)

##### Image transformation and data augmentation (random transformations) #####

# Create a figure
fig = plt.figure(figsize=(14, 12)) # figure size in inches

# For 3 images, randomly apply transformation
for i,sample in enumerate(celeba_train.take(3)): # this step takes a long time (30 min?)
    image = sample['image']

    # Original image
    ax = fig.add_subplot(3, 4, i*4+1)
    ax.imshow(image)
    if i == 0:
        ax.set_title('Orig.', size=15)

    # Random crop (crop to 178 x 178)
    ax = fig.add_subplot(3, 4, i*4+2)
    img_crop = tf.image.random_crop(image, size=(178, 178, 3))
    ax.imshow(img_crop)
    if i == 0:
        ax.set_title('Step 1: Random crop', size=15)

    # Random flip (horizontal)
    ax = fig.add_subplot(3, 4, i*4+3)
    img_flip = tf.image.random_flip_left_right(img_crop)
    ax.imshow(tf.cast(img_flip, tf.uint8))
    if i == 0:
        ax.set_title('Step 2: Random flip', size=15)

    # Resize to 128 x 128
    ax = fig.add_subplot(3, 4, i*4+4)
    img_resize = tf.image.resize(img_flip, size=(128, 128))
    ax.imshow(tf.cast(img_resize, tf.uint8))
    if i == 0:
        ax.set_title('Step 3: Resize', size=15)

plt.show(block=True)

##### Image transformation and data augmentation (random transformations as part of preprocessing of images) #####

# Preprocess image data
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

# Set up training dataset so that it provides 2 images for 5 times
ds = celeba_train.shuffle(1000, reshuffle_each_iteration=False)
ds = ds.take(2).repeat(5)

# Preprocess image data
ds = ds.map(lambda x:preprocess(x, size=(178, 178), mode='train'))

# Display the preprocessed images
fig = plt.figure(figsize=(15, 6)) # figure size in inches
for j,sample in enumerate(ds): # this step takes a long time (2 hours?)
    ax = fig.add_subplot(2, 5, j//2+(j%2)*5+1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(sample[0])
    
plt.show(block=True)
