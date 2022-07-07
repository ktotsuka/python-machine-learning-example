import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.plotting import scatterplotmatrix
import numpy as np
from mlxtend.plotting import heatmap
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RANSACRegressor
from sklearn.model_selection import train_test_split
import scipy as sp
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# Read the housing data
df = pd.read_csv('./housing-data.txt', sep='\s+')
df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 
              'NOX', 'RM', 'AGE', 'DIS', 'RAD', 
              'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

# Set up the input variable, X, and the output variable, y
X = df.iloc[:, :-1].values # grab all rows (samples) and all columns (features) except for the last column (the output)
y = df['MEDV'].values

# Split the data into the training data set and the testing data set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Train the Lasso model (Lasso model has regularization built in)
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)
print(lasso.coef_)

# Predict the output using the trained model
y_train_pred = lasso.predict(X_train)
y_test_pred = lasso.predict(X_test)

# Evaluate the model accuracy
print('MSE train: %.3f, test: %.3f' % (
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_test, y_test_pred)))
print('R^2 train: %.3f, test: %.3f' % (
        r2_score(y_train, y_train_pred),
        r2_score(y_test, y_test_pred)))

# Other models that has regularization built in
ridge = Ridge(alpha=1.0)
elanet = ElasticNet(alpha=1.0, l1_ratio=0.5)