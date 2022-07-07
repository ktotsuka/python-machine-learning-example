import tensorflow as tf
import numpy as np
import pandas as pd
import sklearn
import sklearn.model_selection

# Seed random generators
tf.random.set_seed(1)
np.random.seed(1)

##### Download the MPG data #####

# Set up information for downloading the MPG data
dataset_path = tf.keras.utils.get_file("auto-mpg.data", 
                                       ("http://archive.ics.uci.edu/ml/machine-learning-databases"
                                        "/auto-mpg/auto-mpg.data"))
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower',
                'Weight', 'Acceleration', 'ModelYear', 'Origin']

# Download the data as Panda dataframe
df = pd.read_csv(dataset_path, names=column_names,
                 na_values = "?", comment='\t', # na_values: N/A value
                 sep=" ", skipinitialspace=True)

##### Format the MPG data #####

# Drop samples that has N/A data
print(df.isna().sum()) # print out the number of N/A for each feature column
df = df.dropna() # drop samples that has N/A data
df = df.reset_index(drop=True) # reset sample index so that they will be consecutive again
                               # drop=True: prevent the index to be added as a new column

# Split data into training (80%) and testing (20%) sets
df_train, df_test = sklearn.model_selection.train_test_split(df, train_size=0.8)

# Get statistics (mean, std, etc.) of the training data set
train_stats = df_train.describe().transpose()

# Define names for the features that are numeric (treated as floating point)
numeric_column_names = ['Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration']

# Standardize the numeric features of training and testing data sets
df_train_norm, df_test_norm = df_train.copy(), df_test.copy()
for col_name in numeric_column_names:
    mean = train_stats.loc[col_name, 'mean']
    std  = train_stats.loc[col_name, 'std']
    df_train_norm.loc[:, col_name] = (df_train_norm.loc[:, col_name] - mean)/std
    df_test_norm.loc[:, col_name] = (df_test_norm.loc[:, col_name] - mean)/std

##### Create feature columns #####

# Set up a list of numeric features as Tensorflow feature columns (does not contain data yet)
numeric_features = []
for col_name in numeric_column_names:
    numeric_features.append(tf.feature_column.numeric_column(key=col_name))
    
# Set up a list of bucketized features as Tensorflow feature columns (does not contain data yet)
# There is only one bucketized feature, which is "ModelYear"
# 0: ~ 73
# 1: 73 ~ 76
# 2: 76 ~ 79
# 3: 76 ~
feature_year = tf.feature_column.numeric_column(key="ModelYear")
bucketized_features = []
bucketized_features.append(tf.feature_column.bucketized_column(
    source_column=feature_year,
    boundaries=[73, 76, 79]))

# Set up a list of categorical features as Tensorflow feature columns (does not contain data yet)
# There is only one categorical feature, which is "Origin"
# 1: US
# 2: Europe
# 3: Japan
feature_origin = tf.feature_column.categorical_column_with_vocabulary_list(
    key='Origin',
    vocabulary_list=[1, 2, 3])
categorical_indicator_features = []
categorical_indicator_features.append(tf.feature_column.indicator_column(feature_origin))

# Group together all feature columns
all_feature_columns = (numeric_features + 
                       bucketized_features + 
                       categorical_indicator_features)

##### Input functions for data loading #####

# For training
def train_input_fn(df_train, batch_size=8):
    df = df_train.copy()
    train_x, train_y = df, df.pop('MPG') # train_x: all features, train_y: MPG output
    dataset = tf.data.Dataset.from_tensor_slices((dict(train_x), train_y))
    # For training, we shuffle and repeat the data set
    return dataset.shuffle(len(train_y)).repeat().batch(batch_size)

# For evaluating and predicting
def eval_input_fn(df_test, batch_size=8):
    df = df_test.copy()
    test_x, test_y = df, df.pop('MPG')
    dataset = tf.data.Dataset.from_tensor_slices((dict(test_x), test_y))
    return dataset.batch(batch_size)

##### Define an estimator #####

# Use DNN (deep neural network) regression model
# Use two hidden layers
regressor = tf.estimator.DNNRegressor(
    feature_columns=all_feature_columns,
    hidden_units=[32, 10], # number of units for each hidden layer
    model_dir='models/autompg-dnnregressor/') # directory to save the model (training status)

##### Train the model #####

# Set up parameters for traing the model
EPOCHS = 1000
BATCH_SIZE = 8
total_steps = EPOCHS * int(np.ceil(len(df_train) / BATCH_SIZE))
print('Training Steps:', total_steps)

# Train the model.  It will save the progress at the specified directory
regressor.train(
    input_fn=lambda:train_input_fn(df_train_norm, batch_size=BATCH_SIZE),
    steps=total_steps)

##### Load the model (to illustrate that you can save and load a model) #####

# Load the saved model
reloaded_regressor = tf.estimator.DNNRegressor(
    feature_columns=all_feature_columns,
    hidden_units=[32, 10],
    warm_start_from='models/autompg-dnnregressor/',
    model_dir='models/autompg-dnnregressor/')

##### Evaluate the DNN Regressor model #####

# Evaluate the model using the testing data set
eval_results = reloaded_regressor.evaluate(
    input_fn=lambda:eval_input_fn(df_test_norm, batch_size=BATCH_SIZE))

# Print the fit
# average_loss: mean loss per sample ???
# label/mean: mean of the actual output
# loss: mean loss per mini-batch ???
# prediction/mean: mean of the predicted output
for key in eval_results:
    print('{:15s} {}'.format(key, eval_results[key]))    

##### Predict using the DNN Regressor model #####

# Predict the output for the testing data set.  For prediction, only the input is used
pred_res = regressor.predict(input_fn=lambda: eval_input_fn(df_test_norm, batch_size=BATCH_SIZE))

# Print out the the actual output and the first several predicted output
print(df_test_norm['MPG'].values)
print(next(iter(pred_res)))
print(next(iter(pred_res)))
print(next(iter(pred_res)))


