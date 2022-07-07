import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions

# custom layer
# Normally the linear layer is w*x + b, where w is the weight and b is the bias
# In this custom layer, it is w*(x+e) + b, where e is the noise
# For activation function, use relu (Rectified Linear Unit), which is just max(x,0)
class NoisyLinear(tf.keras.layers.Layer):
    def __init__(self, output_dim, noise_stddev=0.1, **kwargs):
        self.output_dim = output_dim
        self.noise_stddev = noise_stddev
        super(NoisyLinear, self).__init__(**kwargs)

    def build(self, input_shape):
        # Define weights.  The values are set to random
        self.w = self.add_weight(name='weights',
                                 shape=(input_shape[1], self.output_dim),
                                 initializer='random_normal',
                                 trainable=True)
        # Define bias.  The values are set to zero
        self.b = self.add_weight(shape=(self.output_dim,),
                                 initializer='zeros',
                                 trainable=True)

    def call(self, inputs, training=False):
        # If training, add noise.  If not (predicting), don't add noise
        if training:
            batch = tf.shape(inputs)[0]
            dim = tf.shape(inputs)[1]
            noise = tf.random.normal(shape=(batch, dim),
                                     mean=0.0,
                                     stddev=self.noise_stddev)

            noisy_inputs = tf.add(inputs, noise)
        else:
            noisy_inputs = inputs
        z = tf.matmul(noisy_inputs, self.w) + self.b
        return tf.keras.activations.relu(z)
    
    def get_config(self):
        config = super(NoisyLinear, self).get_config()
        config.update({'output_dim': self.output_dim,
                       'noise_stddev': self.noise_stddev})
        return config

# Seed random
np.random.seed(1)
# Commented out the next line because the model always fails to converge
# tf.random.set_seed(1)

# Create input.  Uniformly distributed from -1 to 1.
# Two dimentions (x1 and x2) for 200 samples
x = np.random.uniform(low=-1, high=1, size=(200, 2))

# Create output, which is XOR
y = np.ones(len(x))
y[x[:, 0] * x[:, 1] < 0] = 0

# Split into training and validation data set (100 samples each)
x_train = x[:100, :]
y_train = y[:100]
x_valid = x[100:, :]
y_valid = y[100:]

# Create a nosisy layer
noisy_layer = NoisyLinear(4) # output diemention of 4
noisy_layer.build(input_shape=(None, 4)) # input diemention of 4

# Create a fake input
x = tf.zeros(shape=(1, 4))

# Feed the fake input to the noisy layer and print out the result
# The input, x, is all zeros.  The bias of the layer is all zeros.  But the noise added to the input makes the output non-zero.
# Also, the activation function (relu) makes all non-negative output to 0
tf.print(noisy_layer(x, training=True))

# Save the configuration of the noisiy layer
config = noisy_layer.get_config()

# Create anothe noisy layer using the saved configuration of the original noisy layer
new_layer = NoisyLinear.from_config(config)

# Feed the fake input to the new noisy layer and print out the result
# Note that the result is different from the original noisy layer because of the random noise
tf.print(new_layer(x, training=True))

# Create a model using the noisy linear layer as the first hidden layer
model = tf.keras.Sequential([
    NoisyLinear(4, noise_stddev=0.1), # hidden layer 1
    tf.keras.layers.Dense(units=4, activation='relu'), # hidden layer 2
    tf.keras.layers.Dense(units=4, activation='relu'), # hidden layer 3
    tf.keras.layers.Dense(units=1, activation='sigmoid')]) # output layer
model.build(input_shape=(None, 2)) # input is x1 and x2
model.summary()

# Compile the model
model.compile(optimizer=tf.keras.optimizers.SGD(),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=[tf.keras.metrics.BinaryAccuracy()])

# Train the model
hist = model.fit(x_train, y_train, 
                 validation_data=(x_valid, y_valid), 
                 epochs=200, batch_size=2, 
                 verbose=0)

# Plot loss, accuracy, and fit
# With 3 hidden layers, the fit will be very good
history = hist.history

fig = plt.figure(figsize=(16, 4)) # figure size in inches
ax = fig.add_subplot(1, 3, 1)
plt.plot(history['loss'], lw=4)
plt.plot(history['val_loss'], lw=4)
plt.legend(['Train loss', 'Validation loss'], fontsize=15)
ax.set_xlabel('Epochs', size=15)

ax = fig.add_subplot(1, 3, 2)
plt.plot(history['binary_accuracy'], lw=4)
plt.plot(history['val_binary_accuracy'], lw=4)
plt.legend(['Train Acc.', 'Validation Acc.'], fontsize=15)
ax.set_xlabel('Epochs', size=15)

ax = fig.add_subplot(1, 3, 3)
plot_decision_regions(X=x_valid, y=y_valid.astype(np.integer),
                      clf=model)
ax.set_xlabel(r'$x_1$', size=15)
ax.xaxis.set_label_coords(1, -0.025)
ax.set_ylabel(r'$x_2$', size=15)
ax.yaxis.set_label_coords(-0.025, 1)
plt.show(block=True)
