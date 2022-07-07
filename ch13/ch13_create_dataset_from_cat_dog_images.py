import tensorflow as tf
import numpy as np
import pathlib
import matplotlib.pyplot as plt
import os
import tensorflow_datasets as tfds

# Function to preprocess images
def load_and_preprocess(path, label):
    img_width, img_height = 120, 80
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image) # The image type is int (0 ~ 255)
    image = tf.image.resize(image, [img_height, img_width]) # This will change the image data type from int to float
    image /= 255.0 # For float image data type, the value must be from 0 ~ 1
    return image, label

# Get the list of files of images
imgdir_path = pathlib.Path('cat_dog_images')
file_list = sorted([str(path) for path in imgdir_path.glob('*.jpg')])
print(file_list)

# Display the images
fig = plt.figure(figsize=(10, 5)) # width and height in inches
for i,file in enumerate(file_list):
    img_raw = tf.io.read_file(file)
    img = tf.image.decode_image(img_raw)
    print('Image shape: ', img.shape) # each pixel has 3 values (RGB)
    ax = fig.add_subplot(2, 3, i+1) # 2x3 = 6 images to display
    ax.set_xticks([]); ax.set_yticks([]) # clear x and y ticks
    ax.imshow(img)
    ax.set_title(os.path.basename(file), size=15)
plt.tight_layout()
plt.show(block=True)

# Create label for the images (1 for dog, 0 for cat)
labels = [1 if 'dog' in os.path.basename(file) else 0
          for file in file_list]
print(labels)

# Create a dataset based on the image file names and the label
ds_files_labels = tf.data.Dataset.from_tensor_slices(
    (file_list, labels))
for item in ds_files_labels:
    print(item[0].numpy(), item[1].numpy())

# Preprocess the images
ds_images_labels = ds_files_labels.map(load_and_preprocess)

# Display the preprocessed images
fig = plt.figure(figsize=(10, 5))  # width and height in inches
for i,example in enumerate(ds_images_labels):
    print(example[0].shape, example[1].numpy())
    ax = fig.add_subplot(2, 3, i+1) # 2x3 = 6 images to display
    ax.set_xticks([]); ax.set_yticks([]) # clear x and y ticks
    ax.imshow(example[0])
    ax.set_title('{}'.format(example[1].numpy()), # use the label as the title
                 size=15)    
plt.tight_layout()
plt.show(block=True)
