import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Define a NN model with two layers (one hidden layer and one output layer)
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units=16, activation='relu'))
model.add(tf.keras.layers.Dense(units=32, activation='relu'))

# Build the model
# Layer 1: 4 x 16 (weights) + 16 (bias) = 80 parameters
# Layer 1: 16 x 32 (weights) + 32 (bias) = 544 parameters
model.build(input_shape=(None, 4))
model.summary()

# Print variables of the model
for v in model.variables:
    print('{:20s}'.format(v.name), v.trainable, v.shape)

# Define a NN model with two hidden layers.  This time do more configuration with Keras API
model = tf.keras.Sequential()
model.add(
    tf.keras.layers.Dense(
        units=16, 
        activation=tf.keras.activations.relu,
        kernel_initializer=tf.keras.initializers.GlorotNormal(),
        bias_initializer=tf.keras.initializers.Constant(2.0)
    ))
model.add(
    tf.keras.layers.Dense(
        units=32, 
        activation=tf.keras.activations.sigmoid,
        kernel_regularizer=tf.keras.regularizers.l1
    ))

# Build the model
model.build(input_shape=(None, 4))
model.summary()

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=[tf.keras.metrics.Accuracy(), 
             tf.keras.metrics.Precision(),
             tf.keras.metrics.Recall(),])