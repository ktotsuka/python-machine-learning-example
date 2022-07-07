import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

## Version 1 style
# Tensorflow V1 uses a static graph by default (you have to manually create a graph and add nodes)

# Create a new computation graph
g = tf.Graph()

# Add nodes to the graph
with g.as_default():
    a = tf.constant(1, name='a')
    b = tf.constant(2, name='b')
    c = tf.constant(3, name='c')
    z = 2*(a - b) + c
    
# Evaluate the final node
with tf.compat.v1.Session(graph=g) as sess: # use "compat" to access V1 functionality
    print('Result: z =', sess.run(z)) # Evaluate z (2*(1-2)+3 = 1)
    print('Result: z =', z.eval()) # alternative way

## Version 2 style
# Tensorflow V2 uses a dynamic graph by default (you don't have to manually create a graph)

# Define input nodes
a = tf.constant(1, name='a')
b = tf.constant(2, name='b')
c = tf.constant(3, name='c')

# Evaluate the final node
z = 2*(a - b) + c
tf.print('Result: z =', z)