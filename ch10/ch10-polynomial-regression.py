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

# Create input and output
X = df[['LSTAT']].values
y = df['MEDV'].values

# Train the linear regression model
lr = LinearRegression()
lr = lr.fit(X, y)

# Train the quadratic regression model
qr = LinearRegression()
quadratic = PolynomialFeatures(degree=2)
X_quad = quadratic.fit_transform(X) # Create the input that is (1, x, x^2).  X just contains (x)
qr = qr.fit(X_quad, y)

# Train the cubic regression model
cr = LinearRegression()
cubic = PolynomialFeatures(degree=3)
X_cubic = cubic.fit_transform(X) # Create the input that is (1, x, x^2, x^3).  X just contains (x)
cr = cr.fit(X_cubic, y)

# Create a line for the fit for all three models
X_fit = np.arange(X.min(), X.max(), 1)[:, np.newaxis]
y_lin_fit = lr.predict(X_fit)
y_quad_fit = qr.predict(quadratic.fit_transform(X_fit))
y_cubic_fit = cr.predict(cubic.fit_transform(X_fit))

# Evaluate the three models
linear_r2 = r2_score(y, lr.predict(X))
quadratic_r2 = r2_score(y, qr.predict(X_quad))
cubic_r2 = r2_score(y, cr.predict(X_cubic))

# Plot the fit for all three models
plt.scatter(X, y, label='Training points', color='lightgray')
plt.plot(X_fit, y_lin_fit, 
         label='Linear (d=1), $R^2=%.2f$' % linear_r2, 
         color='blue', 
         lw=2, 
         linestyle=':')
plt.plot(X_fit, y_quad_fit, 
         label='Quadratic (d=2), $R^2=%.2f$' % quadratic_r2,
         color='red', 
         lw=2,
         linestyle='-')
plt.plot(X_fit, y_cubic_fit, 
         label='Cubic (d=3), $R^2=%.2f$' % cubic_r2,
         color='green', 
         lw=2, 
         linestyle='--')
plt.xlabel('% lower status of the population [LSTAT]')
plt.ylabel('Price in $1000s [MEDV]')
plt.legend(loc='upper right')

plt.show()
