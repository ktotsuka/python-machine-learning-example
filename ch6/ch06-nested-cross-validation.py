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

# Combine transformers and estimator (SVC) in a pipeline
pipe_svc = make_pipeline(StandardScaler(),
                         SVC(random_state=1))

# Set up the parameters to explore for the SVC model
param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
param_grid = [{'svc__C': param_range, 
               'svc__kernel': ['linear']},
              {'svc__C': param_range, 
               'svc__gamma': param_range, 
               'svc__kernel': ['rbf']}]

# Perform the grid search to tune the parameter for SVC model (inner loop)
gs = GridSearchCV(estimator=pipe_svc,
                  param_grid=param_grid,
                  scoring='accuracy',
                  cv=2)

# Train the SVC model with the optimal parameter (outer loop)
scores = cross_val_score(gs, X_train, y_train, 
                         scoring='accuracy', cv=5)

# Print out the accuracy of the SVC model
print('CV accuracy for SVC model: %.3f +/- %.3f' % (np.mean(scores),
                                      np.std(scores)))

# Perform the grid search to tune the parameter for decision tree classifier model (inner loop)
gs = GridSearchCV(estimator=DecisionTreeClassifier(random_state=0),
                  param_grid=[{'max_depth': [1, 2, 3, 4, 5, 6, 7, None]}],
                  scoring='accuracy',
                  cv=2)

# Train the decision tree classifier model with the optimal parameter (outer loop)
scores = cross_val_score(gs, X_train, y_train, 
                         scoring='accuracy', cv=5)

# Print out the accuracy of the Decision tree classifier model
print('CV accuracy for decision tree classifier model: %.3f +/- %.3f' % (np.mean(scores), 
                                      np.std(scores)))
