import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
import time
import itertools

# Constants
Z_SIZE = 20
IMAGE_SIZE = (28, 28)
BATCH_SIZE = 128

# Define a function for making the generator
# Input: An array of size 20 with random values.  
#        Either uniform distribution from -1.0 to 1.0, or normal distribution with mean of 0 and standard deviation of 1
# Output: Generated image with size 28 x 28 with data type (-1.0 ~ 1.0)
#
# Input layer: -> 20
# Dense layer: 20 -> 6272
# BatchNorm layer:
# LeakyReLU activation function layer:
# Reshape layer: 6272 -> 7 x 7 x 128
# Conv2DTranspose layer: 7 x 7 x 128 -> 7 x 7 x 128
# BatchNorm layer:
# LeakyReLU activation function layer:
# Conv2DTranspose layer: 7 x 7 x 128 -> 14 x 14 x 64
# BatchNorm layer:
# LeakyReLU activation function layer:
# Conv2DTranspose layer: 14 x 14 x 64 -> 28 x 28 x 32
# BatchNorm layer:
# LeakyReLU activation function layer:
# Conv2DTranspose layer: 28 x 28 x 32 -> 28 x 28 x 1
def make_dcgan_generator(
        z_size=Z_SIZE,
        output_size=(28, 28, 1),
        n_filters=128,
        n_blocks=2):
    # Set up parameters
    size_factor = 2**n_blocks # 2**2 = 4
    hidden_size = (
        output_size[0]//size_factor, # 28 // 4 = 7
        output_size[1]//size_factor # 28 // 4 = 7
    )
    
    # Create a model with an input layer
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(z_size,)), # size
    ])

    # Add layers to the model
    model.add(tf.keras.layers.Dense(
            units=n_filters*np.prod(hidden_size), # 128 * 7 * 7 = 6272
            use_bias=False)) # don't use bias because the BatchNormalization layer handles the bias
    model.add(tf.keras.layers.BatchNormalization()) # standardize the data for better performance of the model (p649)
                                                    # It produces 2 trainable parameters (beta, gamma) and 2 untrainable parameters (mean, std) per output
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Reshape((hidden_size[0], hidden_size[1], n_filters)))
    
    model.add(tf.keras.layers.Conv2DTranspose( # convolutional layer for upsampling
            filters=n_filters, # number of feature maps
            kernel_size=(5, 5),
            strides=(1, 1),
            padding='same', # output size is the same as the input if the stride is 1
            use_bias=False)) # don't use bias because the BatchNormalization layer handles the bias
    model.add(tf.keras.layers.BatchNormalization()) # standardize the data for better performance of the model
    model.add(tf.keras.layers.LeakyReLU())

    # Add more layers to the model
    n_filters_temp = n_filters
    for i in range(n_blocks):
        n_filters_temp = n_filters_temp // 2
        model.add(
            tf.keras.layers.Conv2DTranspose(
                filters=n_filters_temp, # number of feature maps
                kernel_size=(5, 5),
                strides=(2, 2), # since the padding is 'same', the output size will double.
                                # note that the strides for Conv2DTranspose works in opposite way (upsample vs. downsample)
                                # compared to the regular Conv2D
                padding='same', # output size is the same as the input if the stride is 1
                use_bias=False)) # don't use bias because the BatchNormalization layer handles the bias
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.LeakyReLU())

    # Add the final layer
    model.add(
        tf.keras.layers.Conv2DTranspose(
            filters=output_size[2],
            kernel_size=(5, 5), 
            strides=(1, 1),
            padding='same', # output size is the same as the input if the stride is 1
            use_bias=False, 
            activation='tanh'))
        
    return model

# Define a function for making the discriminator
# Input: An image with size 28 x 28 with data type (-1.0 ~ 1.0)
# Output: Logits boolean (0: fake image, 1: real image)
#
# Input layer: -> 28 x 28 x 1
# Conv2D layer: 28 x 28 x 1 -> 28 x 28 x 64
# BatchNorm layer:
# LeakyReLU activation function layer:
# Conv2D layer: 28 x 28 x 64 -> 14 x 14 x 128
# BatchNorm layer:
# LeakyReLU activation function layer:
# Dropout layer:
# Conv2D layer: 14 x 14 x 128 -> 7 x 7 x 256
# BatchNorm layer:
# LeakyReLU activation function layer:
# Dropout layer:
# Conv2D layer: 7 x 7 x 256 -> 1 x 1 x 1
# Reshape layer: 1 x 1 x 1 -> 1
def make_dcgan_discriminator(
        input_size=(28, 28, 1),
        n_filters=64, 
        n_blocks=2):

    # Create a model with an input layer
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_size),
    ])
    
    # Add layers to the model
    model.add(tf.keras.layers.Conv2D(
            filters=n_filters, # number of feature maps
            kernel_size=5, 
            strides=(1, 1),
            padding='same')) # output size is the same as the input if the stride is 1
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    
    # Add more layers to the model
    n_filters_temp = n_filters
    for i in range(n_blocks):
        n_filters_temp = n_filters_temp*2
        model.add(tf.keras.layers.Conv2D(
                filters=n_filters_temp, # number of feature maps
                kernel_size=(5, 5), 
                strides=(2, 2), # since the padding is 'same', the output size will be half.
                padding='same')) # output size is the same as the input if the stride is 1
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.LeakyReLU())
        model.add(tf.keras.layers.Dropout(0.3))
        
    # Add the final layer
    model.add(tf.keras.layers.Conv2D(
            filters=1, # number of feature maps
            kernel_size=(7, 7), # Since the kernel size and the data size is the same, the output size will be 1 x 1
            padding='valid')) # no padding.  
    model.add(tf.keras.layers.Reshape((1,)))
    
    return model

# Function to preprocess the MNIST images, and generate the input to the generator
# Convert image pixcel value from (0 ~ 255) to (-1.0 ~ 1.0) to match with the output of the generator
def preprocess(ex, mode='uniform'):
    # Preprocess image
    image = ex['image']
    image = tf.image.convert_image_dtype(image, tf.float32) # convert from (0 ~ 255) to (0.0 ~ 1.0)
    image = image*2 - 1.0

    # Generate input
    if mode == 'uniform':
        # Get input with uniform distribution from -1 to 1
        input_z = tf.random.uniform(
            shape=(Z_SIZE,), minval=-1.0, maxval=1.0)
    elif mode == 'normal':
        # Get input with normal distribution with mean of 0 and standard deviation of 1
        input_z = tf.random.normal(shape=(Z_SIZE,))
    return input_z, image

# Function to generate sample images using the generator
# The image data type will be 0 ~ 1
def create_samples(g_model, input_z):
    g_output = g_model(input_z, training=False)
    images = tf.reshape(g_output, (BATCH_SIZE, *IMAGE_SIZE))    
    return (images+1)/2.0 # Convert data type from -1.0 ~ 1.0 to 0.0 ~ 1.0