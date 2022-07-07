import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Create Tensorflow variables and print them
a = tf.Variable(initial_value=3.14, name='var_a')
b = tf.Variable(initial_value=[1, 2, 3], name='var_b')
c = tf.Variable(initial_value=[True, False], dtype=tf.bool)
d = tf.Variable(initial_value=['abc'], dtype=tf.string)
print(a)
print(b)
print(c)
print(d)

# Create a non-trainable Tensorflow variable (Tensorflow variables are trainable by default)
w = tf.Variable([1, 2, 3], trainable=False)
print(w.trainable)

# Update the variable value
print(w.assign([3, 1, 4], read_value=True)) # read_value=True: read the updated value
w.assign_add([2, -1, 2], read_value=False) # add value to the variable
print(w.value())

# Create initialization values using Glorot algorithm
# Glorot algorithm was developed by Xavier Glorot and it creates initialization values that is optimized for training model parameters
tf.random.set_seed(1)
init = tf.keras.initializers.GlorotNormal()
tf.print(init(shape=(3,)))
v = tf.Variable(init(shape=(2, 3)))
tf.print(v)

# Custom module inherited from Tensorflow module, which is useful for representing a layer in NN
class MyModule(tf.Module):
    def __init__(self):
        init = tf.keras.initializers.GlorotNormal()
        self.w1 = tf.Variable(init(shape=(2, 3)), trainable=True)
        self.w2 = tf.Variable(init(shape=(1, 2)), trainable=False)

# Instantiate the custom module      
m = MyModule()

# Print variables in the module
print('All module variables: ', [v.shape for v in m.variables])
print('Trainable variable:   ', [v.shape for v in
                                 m.trainable_variables])

# Create some uniformly distributed random variable
tf.random.set_seed(1)
w = tf.Variable(tf.random.uniform((3, 3)))

# Function for matrix multiplication
@tf.function # decoration to tell Tensorflow to create a static computation graph to improve computational efficiency
def compute_z(x):    
    return tf.matmul(w, x)

# Compute result given some input
x = tf.constant([[1], [2], [3]], dtype=tf.float32)
tf.print(compute_z(x))