import tensorflow as tf
import numpy as np
import pathlib
import matplotlib.pyplot as plt
import os
import tensorflow_datasets as tfds

# Create a fake dataset for input, x, and output, y
tf.random.set_seed(1)
t_x = tf.random.uniform([4, 3], dtype=tf.float32) # x: 4x3.  The range is [0 ~ 1), which is the default
t_y = tf.range(4) # y: 4x1

# Create a joint dataset for x and y
ds_joint = tf.data.Dataset.from_tensor_slices((t_x, t_y))
print()
for example in ds_joint:
    print('  x: ', example[0].numpy(), 
          '  y: ', example[1].numpy())

# Shuffle the joint dataset
ds = ds_joint.shuffle(buffer_size=len(t_x)) # buffer_size should be the size of the dataset so that shuffle will be performed for the whole dataset
print()
for example in ds:
    print('  x: ', example[0].numpy(), 
          '  y: ', example[1].numpy())

# Create batches from the dataset
ds = ds_joint.batch(batch_size=3,
                    drop_remainder=False)
batch_x, batch_y = next(iter(ds)) # grab the first batch
print()
print('Batch-x: \n', batch_x.numpy())
print('Batch-y:   ', batch_y.numpy())

# Create batches, then repeat (repeats are useful when running epochs for training models)
ds = ds_joint.batch(3).repeat(count=2)
print()
for i,(batch_x, batch_y) in enumerate(ds):
    print(i, batch_x.shape, batch_y.numpy())

# Create repeats, then batches
ds = ds_joint.repeat(count=2).batch(3)
print()
for i,(batch_x, batch_y) in enumerate(ds):
    print(i, batch_x.shape, batch_y.numpy())

# Shuffle -> batch -> repeat
ds = ds_joint.shuffle(buffer_size=len(t_x)).batch(2).repeat(3)
print()
print('Shuffle -> batch -> repeat')
for i,(batch_x, batch_y) in enumerate(ds):
    print(i, batch_x.shape, batch_y.numpy())

# Shuffle -> batch -> repeat.  Repeat more times to see the trend better.
ds = ds_joint.shuffle(buffer_size=len(t_x)).batch(2).repeat(20)
print()
print('Shuffle -> batch -> repeat (repeat 20 times)')
for i,(batch_x, batch_y) in enumerate(ds):
    print(i, batch_x.shape, batch_y.numpy())

# Batch -> shuffle -> repeat (not good because the selection will be always (0,1) or (2,3))
ds = ds_joint.batch(2).shuffle(buffer_size=len(t_x)).repeat(3)
print()
print('Batch -> shuffle -> repeat')
for i,(batch_x, batch_y) in enumerate(ds):
    print(i, batch_x.shape, batch_y.numpy())

# Batch -> shuffle -> repeat.  Repeat more times to see the trend better.
ds = ds_joint.batch(2).shuffle(buffer_size=len(t_x)).repeat(20)
print()
print('Batch -> shuffle -> repeat (repeat 20 times)')
for i,(batch_x, batch_y) in enumerate(ds):
    print(i, batch_x.shape, batch_y.numpy())
