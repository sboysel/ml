# https://archive.ics.uci.edu/ml/datasets/Covertype
# https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.info

library(randomForest)
library(dplyr)

cov <- read.table('covtype.data.gz', sep = ',')

# V11 - V14 = Wilerness_Area Designation (binary)
# V15 - V54 = Soil_Type (40 binary cols, absence or presence of a certain soil type)

#rename(cov, c('V1'='Elevation', 'V2'='Aspect', 'V3'='Slope', 'V4'='Horizontal_Distance_To_Hydrology', 'V5'='Vertical_Distance_To_Hydrology', 'V6'='Horizontal_Distance_To_Roadways', 'V7'='Hillshade_9am', 'V8'='Hillshade_Noon', 'V9'='Hillshade_3pm', 'V10'='Horizontal_Distance_To_Fire_Points', 'V55'='Cover_Type'))
cov$V55 <- as.factor(cov$V55)

cols <- names(cov)[1:55]
system.time(clf <- randomForest(factor(V55) ~ ., data=cov[,cols], ntree=20, nodesize=5, mtry=9))

table(cov$V55, predict(clf, cov[cols]))
sum(cov$V55==predict(clf, cov[cols])) / nrow(cov)
