## Required libraries
library(dplyr)
library(tidyr)
library(ggplot2)
library(caret)


## Importing data
trainingData <- read.csv(file = "trainingData.csv")
validationData <- read.csv(file = "validationData.csv")


## Exploratory Data Analysis
options(scipen = 999)
summary(trainingData[,c(1:5,521:ncol(trainingData))])
str(trainingData[,c(1:5,521:ncol(trainingData))])

# Converting FLOOR, BUILDINGID, SPACEID, RELATIVEPOSITION, 
# USERID and PHONEID to factor
trainingData[,523:528] <- lapply(trainingData[,523:528], 
                                 as.factor)
str(trainingData[,523:528])

# WAPS intensity distribution
compact_tdata <- gather(trainingData[,1:520])
hist(compact_tdata[compact_tdata$value<100,2], 
     breaks = 100)

# Replacing intesities above -30 for -105
for (i in 1:520) {
  trainingData[trainingData[,i] > -30 ,i] <- -105
}

# Check distribution after replacement
compact_tdata1 <- gather(trainingData[,1:520])
hist(compact_tdata[compact_tdata1$value>-105,2], 
     breaks = 100)

# Visualization
build_plot <- ggplot(trainingData, aes(LONGITUDE, LATITUDE)) + 
  geom_point(aes(color = PHONEID)) + 
  facet_grid(factor(FLOOR, level = c(4:0)) ~ BUILDINGID) +
  theme_bw()

build_plot


## Modelling
# Splitting data into train/test sets
set.seed(123)
indices <- createDataPartition(trainingData$BUILDINGID, 
                               p = .7, 
                               list = F)
train <- trainingData[indices,c(1:520)]
test <- trainingData[-indices,c(1:520)]

# Removing variables with zero variance on train set
zero_var <- apply(train[,1:520], 
                  2, 
                  var)  
train_zv <- train[,zero_var != 0]

# Reducing dimension with PCA
princ.comp <- prcomp(train_zv, 
                     scale. = T)
std_dev <- princ.comp$sdev
pr_var <- pr_dev^2
prop_var <- pr_var/sum(pr_var)

# Scree plot
plot(prop_var, xlab = "Principal Component",
     ylab = "Proportion of Variance Explained",
     type = "b")

# Cumulative Scree plot
plot(cumsum(prop_var), xlab = "Principal Component",
     ylab = "Cumulative Proportion of Variance Explained",
     type = "b")

sum(prop_var[1:131]) # 0.8000691

# Changing train set WAPS for principal components
train.pca <- as.data.frame(princ.comp$x)
train.pca <- train.pca[,1:131]

# Applying PCA to testset
test.pca <- as.data.frame(predict(princ.comp, 
                                  newdata = test))
test.pca <- test.pca[,1:131]

# Adding dependent variables to train and test set
train.pca <- cbind(train.pca, 
                   trainingData[indices,521:524])
test.pca <- cbind(test.pca, 
                  trainingData[-indices,521:524])

# Defining train control for caret models
trCtrol <- trainControl(method = "cv", 
                        number = 3,
                        verboseIter = T)

# Predicting BUILDINGID with knn
BD_knn_model <- train(BUILDINGID ~ ., 
                      data = train.pca[,c(1:131,135)],
                      method = "knn",
                      trControl = trCtrol)
test_bd_knn_prediction <- predict(BD_knn_model, 
                                  newdata = test.pca[,1:131])

BD_knn_results <- confusionMatrix(test_bd_knn_prediction, 
                                  test.pca$BUILDINGID)
BD_knn_results

# Predicting BUILDINGID with RandomForest
BD_rf_model <- train(BUILDINGID ~ ., 
                     data = train.pca[,c(1:131,135)],
                     method = "rf",
                     trControl = trCtrol)
test_bd_rf_prediction <- predict(BD_rf_model, 
                                 newdata = test.pca[,1:131])

BD_rf_results <- confusionMatrix(test_bd_rf_prediction, 
                                 test.pca$BUILDINGID)
BD_rf_results

# Predicting FLOOR with knn
FL_knn_model <- train(FLOOR ~ ., 
                      data = train.pca[,c(1:131,134)],
                      method = "knn",
                      trControl = trCtrol)
test_FL_knn_prediction <- predict(FL_knn_model, 
                                  newdata = test.pca[,1:131])

FL_knn_results <- confusionMatrix(test_FL_knn_prediction, 
                                  test.pca$FLOOR)
FL_knn_results

# Predicting FLOOR with RandomForest
FL_rf_model <- train(FLOOR ~ ., 
                     data = train.pca[,c(1:131,134)],
                     method = "rf",
                     trControl = trCtrol)
test_FL_rf_prediction <- predict(FL_rf_model, 
                                 newdata = test.pca[,1:131])

FL_rf_results <- confusionMatrix(test_FL_rf_prediction, 
                                 test.pca$FLOOR)
FL_rf_results
# Predicting LONGITUDE with knn
LG_knn_model <- train(LONGITUDE ~ ., 
                      data = train.pca[,c(1:131,132)],
                      method = "knn",
                      trControl = trCtrol)
test_LG_knn_prediction <- predict(LG_knn_model, 
                                  newdata = test.pca[,1:131])

LG_knn_results <- postResample(test_LG_knn_prediction, 
                               test.pca$LONGITUDE)
LG_knn_results

# Predicting LONGITUDE with RandomForest
LG_rf_model <- train(LONGITUDE ~ ., 
                     data = train.pca[,c(1:131,132)],
                     method = "rf",
                     trControl = trCtrol)
test_LG_rf_prediction <- predict(LG_rf_model, 
                                 newdata = test.pca[,1:131])

LG_rf_results <- postResample(test_LG_rf_prediction, 
                              test.pca$LONGITUDE)
LG_rf_results

# Predicting LATITUDE with knn
LT_knn_model <- train(LATITUDE ~ ., 
                      data = train.pca[,c(1:131,133)],
                      method = "knn",
                      trControl = trCtrol)
test_LT_knn_prediction <- predict(LT_knn_model, 
                                  newdata = test.pca[,1:131])

LT_knn_results <- postResample(test_LT_knn_prediction, 
                               test.pca$LATITUDE)
LT_knn_results

# Predicting LATITUDE with RandomForest
LT_rf_model <- train(LATITUDE ~ ., 
                     data = train.pca[,c(1:131,133)],
                     method = "rf",
                     trControl = trCtrol)
test_LT_rf_prediction <- predict(LT_rf_model, 
                                 newdata = test.pca[,1:131])

LT_rf_results <- postResample(test_LT_rf_prediction, 
                              test.pca$LATITUDE)
LT_rf_results

# Errors exploration
test_wp <- cbind(test.pca, 
                 test_bd_knn_prediction,
                 test_bd_rf_prediction,
                 test_FL_knn_prediction,
                 test_FL_rf_prediction,
                 test_LG_knn_prediction,
                 test_LG_rf_prediction,
                 test_LT_knn_prediction,
                 test_LT_rf_prediction)

build_knn_errors <- test_wp[test_wp$BUILDINGID != 
                              test_wp$test_bd_knn_prediction,]

floor_knn_errors <- test_wp[test_wp$FLOOR != 
                              test_wp$test_FL_knn_prediction,]

View(floor_knn_errors[,c("BUILDINGID", "FLOOR", "test_FL_knn_prediction")])