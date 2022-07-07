from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import sys
import gzip
import shutil
import os
import struct
import numpy as np
import matplotlib.pyplot as plt

def load_mnist(path, kind='train'):
    """Load MNIST data (image of numbers) from `path`"""
    labels_path = os.path.join(path, '%s-labels.idx1-ubyte' % kind)
    images_path = os.path.join(path, '%s-images.idx3-ubyte' % kind)
        
    with open(labels_path, 'rb') as lbpath:
        # Read the 1st 8 bytes which contains the header information (magic number and the number of data bytes)
        magic, n = struct.unpack('>II', lbpath.read(8)) # '>II':  >: big endien, I: unsigned int number (one for magic number, another for number of data bytes)
        # Read the rest which contains the labels for the images
        labels = np.fromfile(lbpath, dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        # Read the 1st 16 bytes which contains the header information (magic number, etc.)
        magic, num, rows, cols = struct.unpack(">IIII", imgpath.read(16)) # '>IIII':  >: big endien, I: unsigned int number (each of the 4 variables)
                                                                          # magic: magic number
                                                                          # num: number of images
                                                                          # image height
                                                                          # image width
        # Read the rest which contains the image data.  Reshape so that each row corresponds to an image
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)
        # Transform each pixel to have the value from -1 to 1
        images = ((images / 255.) - .5) * 2
 
    return images, labels

# Load the training image data
# X_train contains 60000 flattened image data (each having 28 pixel x 28 pixel --> 784 pixel)
# y_train contains the single digit number that the image represents
X_train, y_train = load_mnist('', kind='train')
print('Rows: %d, columns: %d' % (X_train.shape[0], X_train.shape[1]))

# Load the testing image data
X_test, y_test = load_mnist('', kind='t10k')
print('Rows: %d, columns: %d' % (X_test.shape[0], X_test.shape[1]))

# Visualize the first digit of each class
fig, ax = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True)
ax = ax.flatten() # flatten the subplot from 2x5 to 10x1 to make it easier to iterate them
for i in range(10): # for each digit 0 to 9
    img = X_train[y_train == i][0].reshape(28, 28)
    ax[i].imshow(img, cmap='Greys')
ax[0].set_xticks([]) # remove x ticks
ax[0].set_yticks([]) # remove y ticks
plt.tight_layout()

# Visualize 25 different versions of "7"
fig, ax = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True,)
ax = ax.flatten()
for i in range(25):
    img = X_train[y_train == 7][i].reshape(28, 28)
    ax[i].imshow(img, cmap='Greys')

ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()

plt.show()

# Save the image data as npz (numpy compressed)
np.savez_compressed('mnist_scaled.npz', 
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test)
