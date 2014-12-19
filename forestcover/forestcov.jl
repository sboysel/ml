# Following DecisionTree docs
# https://github.com/bensadeghi/DecisionTree.jl
using DecisionTree
using DataFrame

cov = readtable("covtype.wsv")

# on Julia v0.3
features = array(cov[:, 1:54]);
labels = array(cov[:, 55]);

# train random forest classifier
# using 2 random features, 10 trees, and 0.5 portion of samples per tree (optional)
model = build_forest(labels, features, 20, 30, 0.5)
# apply learned model
apply_forest(model, [5.9,3.0,5.1,1.9])
# run n-fold cross validation for forests
# using 20 random features, 30 trees, 10 folds and 0.5 of samples per tree (optional)
accuracy = nfoldCV_forest(labels, features, 20, 20, 5, 0.5)
