# http://blog.yhathq.com/posts/comparing-random-forests-in-python-and-r.html
library(randomForest)

winered <- read.csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", sep = ";")
winered$color <- rep('red', dim(winered)[1])


winewhite <- read.csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", sep = ";")
winewhite$color <- rep('white', dim(winewhite)[1])

wine <- rbind(winered, winewhite)

wine$is_red <- factor(ifelse(wine$color=='red', 1, 0))
wine$high <- quality <- factor(ifelse(wine$quality > 6, 1, 0))
wine$quality <- factor(wine$quality)

cols <- names(wine)[1:12]

system.time(clf <- randomForest(factor(quality) ~ ., data=wine[,cols], ntree=20, nodesize=5, mtry=9))
 
table(wine$quality, predict(clf, wine[cols]))

sum(wine$quality==predict(clf, wine[cols])) / nrow(wine)
