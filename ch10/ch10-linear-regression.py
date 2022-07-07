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

class LinearRegressionGD(object):
    def __init__(self, eta=0.001, n_iter=20):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1]) # +1 for the offset
        self.cost_ = []

        for i in range(self.n_iter):
            output = self.net_input(X)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return self.net_input(X)

def lin_regplot(X, y, model):
    plt.scatter(X, y, c='steelblue', edgecolor='white', s=70)
    plt.plot(X, model.predict(X), color='black', lw=2)    
    return 

# Read the housing data
df = pd.read_csv('./housing-data.txt', sep='\s+')
df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 
              'NOX', 'RM', 'AGE', 'DIS', 'RAD', 
              'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

# Create scatter plots to see how the data is distributed
cols = ['LSTAT', 'INDUS', 'NOX', 'RM', 'MEDV'] # Plot for these data only
scatterplotmatrix(df[cols].values, figsize=(10, 8), 
                  names=cols, alpha=0.5)
plt.tight_layout()

# Plot heatmap to show correlation between variables (-1: perfect negative corelation, 0: no corelation, 1: perfect positive corelation)
cm = np.corrcoef(df[cols].values.T) # T = transposed data
hm = heatmap(cm, row_names=cols, column_names=cols)

# Set up a single input and a single output
X = df[['RM']].values
y = df['MEDV'].values

# Standardized the input and the output
sc_x = StandardScaler()
sc_y = StandardScaler()
X_std = sc_x.fit_transform(X)
y_std = sc_y.fit_transform(y[:, np.newaxis]).flatten() # fit_transform() needs an array as input, so convert it to an array, then change it back to flattened data

# Train the model
lr = LinearRegressionGD()
lr.fit(X_std, y_std)

# Plot the error vs. training iteration
plt.figure(3)
plt.plot(range(1, lr.n_iter+1), lr.cost_)
plt.ylabel('SSE')
plt.xlabel('Epoch')

# Plot the predicted and actual data
plt.figure(4)
lin_regplot(X_std, y_std, lr)
plt.xlabel('Average number of rooms [RM] (standardized)')
plt.ylabel('Price in $1000s [MEDV] (standardized)')

# Print the weights for the linear regression
print('Slope: %.3f' % lr.w_[1])
print('Intercept: %.3f' % lr.w_[0]) # This is 0 because the data is standardized

# Print the output in its original scale ($).  Use an example input of 5 (5 rooms)
num_rooms_std = sc_x.transform(np.array([[5.0]])) # standardided input
price_std = lr.predict(num_rooms_std) # standardized output
print("Price in $1000s: %.3f" % sc_y.inverse_transform(price_std.reshape(1, -1))) # output in original scalse

plt.show()

