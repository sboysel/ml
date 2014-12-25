# http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html 
# https://www.kaggle.com/c/titanic-gettingStarted/details/getting-started-with-random-forests

import math
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

cov = pd.read_csv('covtype.wsv', sep = ' ') 

# Models
model_baseline = RandomForestClassifier()
model1 = RandomForestClassifier(n_estimators = 25,
                                max_features = 30,
                                n_jobs = -1,
                                verbose = 1)

# Feature / Label Subsets
X = np.array(cov.ix[:,:54])
y = np.array(cov[[54]])

# Test / Train Split
n = math.floor(2/3 * X.shape[0])
X_train = X[:n,]
y_train = y[:n,]
X_test = X[n:,]
y_test = y[n:,]

# Fit / Predict
model_baseline.fit(X_train, y_train)
model_baseline.predict(X_test)

# Benchmarks:
# model_baseline.score(X_test, y_test)
# Defaults:
# 0.70827180255073063

