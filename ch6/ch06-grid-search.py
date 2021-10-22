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

# Combine transformers and estimator in a pipeline
pipe_svc = make_pipeline(StandardScaler(),
                         SVC(random_state=1))

# Perform grid search using scikit-learn tool
param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

param_grid = [{'svc__C': param_range, 
               'svc__kernel': ['linear']},
              {'svc__C': param_range, 
               'svc__gamma': param_range, 
               'svc__kernel': ['rbf']}]

gs = GridSearchCV(estimator=pipe_svc, 
                  param_grid=param_grid, 
                  scoring='accuracy', 
                  refit=True, # automatically fit the best estimater
                  cv=10, # number of folds for cross validation
                  n_jobs=-1)
gs = gs.fit(X_train, y_train)

# Print the parameter that gives the best accuracy score for the traning data
print(gs.best_score_)
print(gs.best_params_)

# Print out the accuracy of the best estimater for the testing data
# No need to fit the best estimator because we used "refit=True"
clf = gs.best_estimator_
print('Test accuracy: %.3f' % clf.score(X_test, y_test))