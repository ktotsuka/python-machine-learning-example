import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
tf.random.set_seed(1)
np.random.seed(1)

# Create the XOR data
# Inputs: x1 and x2 (-1 ~ 1)
# Output: y = 0 or 1
x = np.random.uniform(low=-1, high=1, size=(200, 2)) # 200 samples
y = np.ones(len(x))
y[x[:, 0] * x[:, 1]<0] = 0

# Devide the data into training and testing data set (100 samples each)
x_train = x[:100, :]
y_train = y[:100]
x_test = x[100:, :]
y_test = y[100:]

##### Define the input functions #####

# For training
def train_input_fn(x_train, y_train, batch_size=8):
    dataset = tf.data.Dataset.from_tensor_slices(
        ({'input-features':x_train}, y_train.reshape(-1, 1)))
    # For training, we shuffle and repeat the data set
    return dataset.shuffle(100).repeat().batch(batch_size)

# For testing
def eval_input_fn(x_test, y_test=None, batch_size=8):
    if y_test is None:
        dataset = tf.data.Dataset.from_tensor_slices(
            {'input-features':x_test})
    else:
        dataset = tf.data.Dataset.from_tensor_slices(
            ({'input-features':x_test}, y_test.reshape(-1, 1)))
    return dataset.batch(batch_size)
    
##### Create the estimator: convert from a Keras model #####

# Define a Keras model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(2,), name='input-features'), # input layer
    tf.keras.layers.Dense(units=4, activation='relu'), # hidden layer 1
    tf.keras.layers.Dense(units=4, activation='relu'), # hidden layer 2
    tf.keras.layers.Dense(units=4, activation='relu'), # hidden layer 3
    tf.keras.layers.Dense(1, activation='sigmoid') # output layer
])
model.summary()

# Compile the model
model.compile(optimizer=tf.keras.optimizers.SGD(),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=[tf.keras.metrics.BinaryAccuracy()]) # Use the binary accuracy as the metric

# Conver the Keras model to an estimator
# The advantage of converting to an estimator is that it provides more functionality
# such as distributed training and automatically saving the checkpoints during training
my_estimator = tf.keras.estimator.model_to_estimator(
    keras_model=model,
    model_dir='models/estimator-for-XOR/') # directory to save the model (training status)

##### Use the estimator: train/evaluate #####

# Define training parameters
num_epochs = 200
batch_size = 2
steps_per_epoch = np.ceil(len(x_train) / batch_size)

# Train the estimator
my_estimator.train(
    input_fn=lambda: train_input_fn(x_train, y_train, batch_size),
    steps=num_epochs * steps_per_epoch)

# Evaluate the estimator
eval_results = my_estimator.evaluate(input_fn=lambda: eval_input_fn(x_test, y_test, batch_size))

# Print the fit
print(eval_results['binary_accuracy'])   