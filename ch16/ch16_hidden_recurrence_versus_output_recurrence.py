import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import pandas as pd
import os
import gzip
import shutil
from collections import Counter
from tensorflow.keras.layers import Embedding
from tensorflow.keras import Sequential
from tensorflow.keras.layers import SimpleRNN
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import GRU
from tensorflow.keras.layers import Bidirectional

# Create a simple RNN model using Keras API
# The model uses output-to-output recurrence
# h(t) = activate(x(t) * w_xh + bh)
# o(t) = activate(h(t) * w_ho + o(t-1)*w_oo + bo)
#
# w_xh: weight for input to hidden layer
# b_h: bias for hidden layer
# w_ho: weight for hidden to output layer
# w_oo: weight for output (at t-1) to output (at t)
# b_o: bias for output layer
#
# The model does not use activation function for the hidden layer and uses tanh for the output layer
# The model uses identity matrix for w_ho
# The model does not have a bias for the output layer
# The simplified equations will be:
#
# h(t) = x(t) * w_xh + bh
# o(t) = tanh(h(t) + o(t-1)*w_oo)
#
# With dimentions
#
# h(t) (1 x 2) = x(t) (1 x 5) * w_xh (5 x 2) + bh (1 x 2)
# o(t) (1 x 2) = tanh(h(t) (1 x 2) + o(t-1) (1 x 2) * w_oo (2 x 2))
rnn_layer = tf.keras.layers.SimpleRNN(
    units=2, use_bias=True, 
    return_sequences=True)
rnn_layer.build(input_shape=(None, None, 5)) # (batch dimention, sequence dimention, feature dimention) (None = unspecified)    

# Get weights and biases
w_xh, w_oo, b_h = rnn_layer.weights
print('W_xh shape:', w_xh.shape)
print('W_oo shape:', w_oo.shape)
print('b_h shape:', b_h.shape)

# Create input (3 (sequence) x 5 (feature))
x_seq = tf.convert_to_tensor(
    [[1.0]*5, [2.0]*5, [3.0]*5],
    dtype=tf.float32)

# Compute the output (3 (sequence) x 2 (unit)) using the simeple RNN Keras API
output = rnn_layer(tf.reshape(x_seq, shape=(1, 3, 5)))

# Compute the output manually
out_man = []
for t in range(len(x_seq)):
    # Get x(t)
    xt = tf.reshape(x_seq[t], (1, 5))
    print('Time step {} =>'.format(t))
    print('   Input (x(t))          :', xt.numpy())
    
    # Compute h(t)
    ht = tf.matmul(xt, w_xh) + b_h    
    print('   Hidden (h(t))      :', ht.numpy())
    
    # Compute o(t-1)
    if t>0:
        prev_o = out_man[t-1]
    else:
        prev_o = tf.zeros(shape=(ht.shape))

    # Comput o(t)        
    ot = ht + tf.matmul(prev_o, w_oo)
    ot = tf.math.tanh(ot)

    # Append o(t) to the output sequence
    out_man.append(ot)

    # Print out o(t) for the manual and simple RNN
    print('   Output (manual) :', ot.numpy())
    print('   SimpleRNN output:'.format(t), output[0][t].numpy())
    print()

