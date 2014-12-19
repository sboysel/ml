using DecisionTree
using DataFrames

# DataFrames notes:
# describe(winered)
# size(winered)
# names(winred)

# curl -O http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv
winered = readtable("winequality-red.csv", separator = ';')
# curl -O http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv
winewhite = readtable("winequality-white.csv", separator = ';')

# Add labels
winered = hcat(winered, ones(size(winered, 1)))
winewhite = hcat(winewhite, zeros(size(winewhite, 1)))

# Combine white and red and rename index
wine = vcat(winewhite, winered)
relabel = names(wine)
relabel[13] = symbol("is_red")
names!(wine, relabel)

features = array(wine[:, 1:12]);
labels = array(wine[:, 13]);

# train random forest classifier
# using 2 random features, 10 trees, and 0.5 portion of samples per tree (optional)
#model = build_forest(labels, features, 6, 30, 0.5)
# apply learned model
#apply_forest(model, [5.9,3.0,5.1,1.9])
# run n-fold cross validation for forests
# using 20 random features, 30 trees, 10 folds and 0.5 of samples per tree (optional)
@time accuracy = nfoldCV_forest(labels, features, 5, 30, 3, 0.5)

# Parallel Comparison:
# julia -p 4
# time = 270.904031385 seconds (458781176 bytes allocated, 0.22% gc time) 
# julia -p 2
# time = 341.157307 seconds (302129248 bytes allocated, 0.14% gc time) 
# julia
# time = 599.956209925 seconds (127471045984 bytes allocated, 31.74% gc time) 
