import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
import time
import itertools
from ch17lib import make_generator_network
from ch17lib import make_discriminator_network
from ch17lib import preprocess
from ch17lib import Z_SIZE
from ch17lib import IMAGE_SIZE
from ch17lib import BATCH_SIZE

##### Build the generator and the discriminator #####

# Set up parameters
gen_hidden_layers = 1
gen_hidden_size = 100
disc_hidden_layers = 1
disc_hidden_size = 100

# Build the generator
# input: z, which contains random values
# output: a generated image of size 28 x 28
gen_model = make_generator_network(
    num_hidden_layers=gen_hidden_layers, 
    num_hidden_units=gen_hidden_size,
    num_output_units=np.prod(IMAGE_SIZE))
gen_model.build(input_shape=(None, Z_SIZE))
gen_model.summary()

# Build the discriminator
# input: an image of size 28 x 28
# output: boolean prediction with logits (0: fake image, 1: real image)
disc_model = make_discriminator_network(
    num_hidden_layers=disc_hidden_layers,
    num_hidden_units=disc_hidden_size)
disc_model.build(input_shape=(None, np.prod(IMAGE_SIZE)))
disc_model.summary()

##### Fetch the MNIST hand-written digit dataset #####

# Fetch the MNIST hand-written digit dataset
mnist_bldr = tfds.builder('mnist')
mnist_bldr.download_and_prepare()
mnist = mnist_bldr.as_dataset()

# Get training dataset for MINST
# It contains images of size 28 x 28
mnist_train_set = mnist['train']

# Print out the image format before preprocessing
print('Before preprocessing:  ')
example_image = next(iter(mnist_train_set))['image']
print('data type: ', example_image.dtype, ' Min: {} Max: {}'.format(np.min(example_image), np.max(example_image)))

# Preprocess the image
mnist_train_set = mnist_train_set.map(preprocess)

# Print out the image format after preprocessing
print('After preprocessing:  ')
example_image = next(iter(mnist_train_set))[0]
print('data type: ', example_image.dtype, ' Min: {} Max: {}'.format(np.min(example_image), np.max(example_image)))

##### Get the output of the untrained generator and the discriminator #####

# Get input, z, and the real images (batch size of 64) from the training dataset
# input_z: Input for the generator.  The shape is (64, 20): Each input has size 20 and there are 64 samples
# input-real: Real images.  The shape is (64, 784): Each image has 784 pixels (28 x 28) and there are 64 samples 
mnist_train_set = mnist_train_set.batch(BATCH_SIZE, drop_remainder=True)
input_z, input_real = next(iter(mnist_train_set))
print('input-z -- shape:', input_z.shape)
print('input-real -- shape:', input_real.shape)

# Get generated images
# g_output: Output of the generator.  The shape is (64, 784).  Each generated image has 784 pixels (28 x 28) and there are 64 samples 
g_output = gen_model(input_z)
print('Output of G -- shape:', g_output.shape)

# Get output of the discriminator using the real images
# d_logits_real: Output of the discriminator using real images.  The shape is (64, 1).  
#                0 (fake image) or 1 (real image) with logits format and there are 64 samples 
d_logits_real = disc_model(input_real)
print('Disc. (real) -- shape:', d_logits_real.shape)

# Get output of the discriminator using the fake images
# d_logits_real: Output of the discriminator using fake images.  The shape is (64, 1).  
#                0 (fake image) or 1 (real image) with logits format and there are 64 samples 
d_logits_fake = disc_model(g_output)
print('Disc. (fake) -- shape:', d_logits_fake.shape)

##### Compute losss for the generator and the discriminator #####

# Instantiate loss function for binary output
loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# Loss for the generator
g_labels_real = tf.ones_like(d_logits_fake)
g_loss = loss_fn(y_true=g_labels_real, y_pred=d_logits_fake)
print('Generator Loss: {:.4f}'.format(g_loss))

# Loss for the discriminator
d_labels_real = tf.ones_like(d_logits_real)
d_labels_fake = tf.zeros_like(d_logits_fake)

d_loss_real = loss_fn(y_true=d_labels_real, y_pred=d_logits_real)
d_loss_fake = loss_fn(y_true=d_labels_fake, y_pred=d_logits_fake)
print('Discriminator Losses: Real {:.4f} Fake {:.4f}'.format(d_loss_real.numpy(), d_loss_fake.numpy()))
