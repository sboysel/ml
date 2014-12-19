# https://archive.ics.uci.edu/ml/datasets/Covertype
# https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.info

library(randomForest)
library(dplyr)

cov <- read.table('covtype.data.gz', sep = ',')

cov$V55 <- as.factor(cov$V55)

cols <- names(cov)[1:55]
system.time(clf <- randomForest(factor(V55) ~ ., data=cov[,cols], ntree=20, nodesize=5, mtry=9))

table(cov$V55, predict(clf, cov[cols]))
sum(cov$V55==predict(clf, cov[cols])) / nrow(cov)
