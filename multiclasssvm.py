# http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
# http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html#sklearn.datasets.load_iris
# 
# http://astrostatistics.psu.edu/datasets/HIP_star.html
# 
import numpy as np
import pandas as pd
from sklearn import datasets, svm, cross_validation

#iris = datasets.load_iris()
hip = pd.read_table( "http://astrostatistics.psu.edu/datasets/HIP_star.dat", delim_whitespace=True)

featuresHIP = hip.drop(["HIP","B-V"], axis = 1)
labelsHIP = hip["B-V"]

featuresHIP_np = np.array(featuresHIP)
labelsHIP_np = np.array(labelsHIP)

#pred1 = clf.predict_proba(xTest)

#xTrain, xTest, yTrain, yTest = cross_validation.train_test_split(iris.data, iris.target, test_size=0.4, random_state=0)

#clf = svm.SVC(probability = True, kernel='linear', C=1).fit(xTrain, yTrain)
#clf.score(xTest, yTest)

# HIP
xTrainHIP, xTestHIP, yTrainHIP, yTestHIP = cross_validation.train_test_split(featuresHIP_np, labelsHIP_np, test_size=0.4, random_state=0)

clf2 = svm.SVC(probability = True, kernel='linear', C=1).fit(xTrainHIP,
        yTrainHIP)
clf2.score(xTestHIP, yTestHIP)

