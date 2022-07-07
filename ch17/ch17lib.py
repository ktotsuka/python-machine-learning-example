import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
import time
import itertools

# Constants
Z_SIZE = 20
IMAGE_SIZE = (28, 28)
BATCH_SIZE = 64

# Define a function for making the generator
# Input: An array of size 20 with random values.  
#        Either uniform distribution from -1.0 to 1.0, or normal distribution with mean of 0 and standard deviation of 1
# Output: Generated image with size 28 x 28 with data type (-1.0 ~ 1.0)
def make_generator_network(
        num_hidden_layers=1,
        num_hidden_units=100,
        num_output_units=784):
    model = tf.keras.Sequential()
    for i in range(num_hidden_layers):
        model.add(
            tf.keras.layers.Dense(
                units=num_hidden_units, 
                use_bias=False)
            )
        # Activation function can be added as a separate layer or an argument to the last layer
        # LeakyReLU is the same as ReLU except it has gradient for negative input values (see page 633)
        model.add(tf.keras.layers.LeakyReLU()) # 
        
    # Create the output layer with tanh activation function whose output is -1 ~ 1
    model.add(tf.keras.layers.Dense(units=num_output_units, activation='tanh'))
    return model

# Define a function for making the discriminator
# Input: An image with size 28 x 28 with data type (-1.0 ~ 1.0)
# Output: Logits boolean (0: fake image, 1: real image)
def make_discriminator_network(
        num_hidden_layers=1,
        num_hidden_units=100,
        num_output_units=1):
    model = tf.keras.Sequential()
    for i in range(num_hidden_layers):
        model.add(tf.keras.layers.Dense(units=num_hidden_units))
        model.add(tf.keras.layers.LeakyReLU())
        model.add(tf.keras.layers.Dropout(rate=0.5))
        
    model.add(
        tf.keras.layers.Dense(
            units=num_output_units, 
            activation=None)
        )
    return model

# Function to preprocess the MNIST images, and generate the input to the generator
# Convert image pixcel value from (0 ~ 255) to (-1.0 ~ 1.0) to match with the output of the generator
def preprocess(ex, mode='uniform'):
    # Preprocess image
    image = ex['image']
    image = tf.image.convert_image_dtype(image, tf.float32) # convert from (0 ~ 255) to (0.0 ~ 1.0)
    image = tf.reshape(image, [-1])
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