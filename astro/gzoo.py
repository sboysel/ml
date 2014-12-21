# http://data.galaxyzoo.org/
# http://data.galaxyzoo.org/data/gz2/zoo2MainSpecz.txt
#
# > wget http://zooniverse-data.s3.amazonaws.com/galaxy-zoo-2/zoo2MainSpecz.csv.gz
#
# Unzip dataset:
# > gunzip zoo2MainSpecz.csv.gz
#
# Citation:
# @article{willett2013galaxy,
# title={Galaxy Zoo 2: detailed morphological classifications for 304 122 galaxies from the Sloan Digital Sky Survey},
# author={Willett, Kyle W and Lintott, Chris J and Bamford, Steven P and Masters, Karen L and Simmons, Brooke D and Casteels, Kevin RV and Edmondson, Edward M and Fortson, Lucy F and Kaviraj, Sugata and Keel, William C and others},
# journal={Monthly Notices of the Royal Astronomical Society},
# pages={stt1458},
# year={2013},
# publisher={Oxford University Press}
# }

import time
import math
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

zoo = pd.read_csv('zoo2MainSpecz.csv')

model_baseline = RandomForestClassifier()

# Select Features
features_list = [3,4]
features_list.extend(np.arange(9, 233))

# Features / Labels Split
X = np.array(zoo.ix[:,features_list])
y = np.ravel(zoo[[8]])

# Test / Train Split
n = math.floor(2/3 * X.shape[0])
X_train = X[:n,]
y_train = y[:n,]
X_test = X[n:,]
y_test = y[n:,]

# Fit / Predict / Score
t1 = time.time()
clf = model_baseline.fit(X_train, y_train)
pred = model_baseline.predict(X_test)
score = model_baseline.score(X_test, y_test)
conf_mat = confusion_matrix(y_test, pred)
t2 = time.time()
elapsed = t2-t1
print("Time Elasped:", elapsed)

print("Accuracy and Confusion Matrix:")
print(score)
conf_mat
