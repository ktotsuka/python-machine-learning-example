import pandas as pd
from io import StringIO
import sys
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.base import clone
from itertools import combinations
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

# Create an array with missing values
csv_data = '''A,B,C,D
1.0,2.0,3.0,4.0
5.0,6.0,,8.0
10.0,11.0,12.0,'''

# Read the data into Panda's dataframe
df = pd.read_csv(StringIO(csv_data))

# Get number of missing values per column
df.isnull().sum()

# Access internal Numpy's array
df.values

# Remove rows that contain missing values
df.dropna(axis=0)

# Remove columns that contain missing values
df.dropna(axis=1)

# Drop rows where all columns are NaN
df.dropna(how='all')  

# Drop rows that have fewer than 4 real values 
df.dropna(thresh=4)

# Drop rows where NaN appear in specific columns (here: 'C')
df.dropna(subset=['C'])

# Impute missing values via the column mean
imr = SimpleImputer(missing_values=np.nan, strategy='mean')
imr = imr.fit(df.values)
imputed_data = imr.transform(df.values)
imputed_data

# Same but using Panda's function
df.fillna(df.mean())