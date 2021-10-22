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
from matplotlib.colors import ListedColormap

def get_data():  
    # Read in the data
    df = pd.read_csv('wdbc.data', header=None)

    # Get inputs and output
    X = df.loc[:, 2:].values
    y = df.loc[:, 1].values

    # Encode the output into numbers
    le = LabelEncoder()
    y = le.fit_transform(y)

    # Split data into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=0.20,
                                                        stratify=y,
                                                        random_state=1)

    return X_train, X_test, y_train, y_test
