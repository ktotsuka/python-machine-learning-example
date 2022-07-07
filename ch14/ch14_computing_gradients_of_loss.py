import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Model: z = w*x + b
# x: input
# w: weight
# b: bias
# z: predicted output
# y: output
#
# loss: (y-z)^2

# Create w and b as TF variables
w = tf.Variable(1.0)
b = tf.Variable(0.5)
print(w.trainable, b.trainable)

# Create input and output as tensors
x = tf.convert_to_tensor([1.4])
y = tf.convert_to_tensor([2.1])

## Compute gradient with respect to trainable tensors

# Record the computation of predicted output and the loss in a 'tape' so that we can compute the gradient later
with tf.GradientTape() as tape:
    z = tf.add(tf.multiply(w, x), b)
    loss = tf.square(y - z)

# Compute the gradient of loss wrt w
dloss_dw = tape.gradient(loss, w)
tf.print('dL/dw : ', dloss_dw)

# You can calculate the gradient manually and the equation will be 2*x * (w*x + b - y)
tf.print(2*x * (w*x + b - y))

## Compute gradient with respect to non-trainable tensors

# Record the computation of predicted output and the loss in a 'tape' so that we can compute the gradient later
# This time, we will compute gradient wrt x, which is not trainable, so we need to "watch" it
with tf.GradientTape() as tape:
    tape.watch(x)
    z = tf.add(tf.multiply(w, x), b)
    loss = tf.square(y - z)

# Compute the gradient of loss wrt x
dloss_dx = tape.gradient(loss, x)
tf.print('dL/dx:', dloss_dx)

# You can calculate the gradient manually and the equation will be 2*w * (w*x + b - y)
tf.print(2*w * (w*x + b - y))

## Compute gradients for multiple tensors while keeping resources

# Record the computation of predicted output and the loss in a 'tape' so that we can compute the gradient later
with tf.GradientTape(persistent=True) as tape: # persistent=True makes the resource for "tape" persistent after each use
    z = tf.add(tf.multiply(w, x), b)
    loss = tf.square(y - z)

# Compute the gradient of loss wrt w and b
dloss_dw = tape.gradient(loss, w) # if persistent=False, the resource for "tape" is released after this line 
dloss_db = tape.gradient(loss, b)
tf.print('dL/dw:', dloss_dw)
tf.print('dL/db:', dloss_db)

# Update parameter based on their gradient
optimizer = tf.keras.optimizers.SGD()
optimizer.apply_gradients(zip([dloss_dw, dloss_db], [w, b]))
tf.print('Updated w:', w)
tf.print('Updated bias:', b)