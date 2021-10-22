import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import make_scorer
from sklearn.metrics import roc_curve, auc
from distutils.version import LooseVersion as Version
from scipy import __version__ as scipy_version
from numpy import interp
from scipy import interp
from sklearn.utils import resample
from ch06lib import *

# Get breast cancer data
X_train, X_test, y_train, y_test = get_data()

# Create imbalanced dataset
X_imb = np.vstack((X_train[y_train == 0], X_train[y_train == 1][:40]))
y_imb = np.hstack((y_train[y_train == 0], y_train[y_train == 1][:40]))

# Show that the majority class estimator can achieve very high accuracy
y_pred = np.zeros(y_imb.shape[0])
print('Accuracy for the majority class estimator: ', np.mean(y_pred == y_imb) * 100)

# Upsample the minority class 
print('Number of class 1 examples before:', X_imb[y_imb == 1].shape[0])
X_upsampled, y_upsampled = resample(X_imb[y_imb == 1],
                                    y_imb[y_imb == 1],
                                    replace=True,
                                    n_samples=X_imb[y_imb == 0].shape[0],
                                    random_state=123)
print('Number of class 1 examples after:', X_upsampled.shape[0])

# Stack the upsampled class to the majority class to create a balanced data
X_bal = np.vstack((X_train[y_train == 0], X_upsampled))
y_bal = np.hstack((y_train[y_train == 0], y_upsampled))

# Now the majority class estimator has only 50% accuracy
y_pred = np.zeros(y_bal.shape[0])
print('Accuracy for the majority class estimator: ', np.mean(y_pred == y_bal) * 100)
