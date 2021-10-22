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
pipe_lr = make_pipeline(StandardScaler(),
                        PCA(n_components=2),
                        LogisticRegression(random_state=1, solver='lbfgs'))

# Split training data into iteration of "folds"
kfold = StratifiedKFold(n_splits=10).split(X_train, y_train)

# Print accuracy for each iteration of folds
scores = []
for k, (train, test) in enumerate(kfold):
    pipe_lr.fit(X_train[train], y_train[train]) # train the model using the traing fold
    score = pipe_lr.score(X_train[test], y_train[test]) # score the accuracy using the test fold
    scores.append(score)
    print('Fold: %2d, Class distribution: %s, Acc: %.3f' % (k+1,
          np.bincount(y_train[train]), score)) # Class distribution is the same for all iteration becaused we used "StratifiedKFold"
    
print('\nMean cross validation accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))

# Do the same but using scikit-learn
scores = cross_val_score(estimator=pipe_lr,
                         X=X_train,
                         y=y_train,
                         cv=10,
                         n_jobs=1)
print('Cross validation accuracy scores: %s' % scores)
print('Mean cross validation accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
