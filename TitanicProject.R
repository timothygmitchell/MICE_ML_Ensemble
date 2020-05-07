################################# READ IN DATA ##########################################

setwd("/home/tim/Documents/R/STAT488Nonparametric/Titanic Project")

library(VIM) # matrix plots
library(miceRanger) # multiple imputation with chained equations
library(randomForest) # random forests
library(gbm) # gradient boosted machines

train <- read.csv("train.csv", stringsAsFactors=T, na.strings = c(NA, "")) # training data
test <- read.csv("test.csv", stringsAsFactors=T, na.strings = c(NA, "")) # testing data

# Treat response variable as factor
train$Survived <- factor(train$Survived)

# Extract titles in training data, keep the most common of these
train$Title <- gsub('(.*, )|(\\..*)', '', train$Name)
train$Title[train$Title == "Capt"] <- "Mr"
train$Title[train$Title == "Col"] <- "Mr"
train$Title[train$Title == "Don"] <- "Mr"
train$Title[train$Title == "Jonkheer"] <- "Mr"
train$Title[train$Title == "Major"] <- "Mr"
train$Title[train$Title == "Sir"] <- "Mr"
train$Title[train$Title == "Lady"] <- "Miss"
train$Title[train$Title == "Mlle"] <- "Miss"
train$Title[train$Title == "Ms"] <- "Miss"
train$Title[train$Title == "Mme"] <- "Mrs"
train$Title[train$Title == "the Countess"] <- "Mrs"
train$Title <- factor(train$Title)

# Extract titles in testing data, standardize levels attribute
test$Title <- gsub('(.*, )|(\\..*)', '', test$Name)
test$Title[test$Title == "Dona"] <- "Mrs"
test$Title[test$Title == "Ms"] <- "Miss"
test$Title[test$Title == "Col"] <- "Mr"
test$Title <- factor(test$Title, levels = levels(train$Title))

# Subset data
train <- within(train, rm("PassengerId","Name","Ticket","Cabin"))
test <- within(test, rm("PassengerId","Name","Ticket","Cabin"))

################################# IMPUTATION #############################################

# Examine missingness
sapply(train, function(x)sum(is.na(x)))
sapply(test, function(x)sum(is.na(x)))

# Does data appear to be missing at random (MAR)?
VIM::matrixplot(train, sortby = "Survived", main = "Missingness in Training Data")
VIM::matrixplot(test, main = "Missingness in Test Data")

# m = 20 multiple imputations
imputeddat <- completeData(miceRanger(train, 
                                      vars = list(Age = c("Pclass", "Sex", "Title"),
                                                  Embarked = c("Pclass", "Sex", "Title")),
                                      m = 20, maxiter = 10))

# m = 20 multiple imputations
imputedtestdat <- completeData(miceRanger(test, 
                                          vars = list(Age = c("Pclass", "Sex", "Title"),
                                                      Fare = c("Pclass", "Sex", "Title")),
                                          m = 20, maxiter = 10))

################################### RANDOM FORESTS ########################################

set.seed(1234)
imputedrfs <- list()

# Generate 20 random forests from 20 imputed data sets
for  (m in 1:20){
  imputedrfs[[m]] <- randomForest(Survived ~ ., data = imputeddat[[m]], 
                                  replace=F, sampsize=750, mtry=2, ntree=500)
}

# Combine random forests
body(combine)[[4]] <- substitute(rflist <- (...)); rf.all <- combine(imputedrfs)
rf.all$importance # Most important variables

# Pool votes to make final predictions
votes <- list()

for  (m in 1:20){
  votes[[m]] <- predict(rf.all, imputedtestdat[[m]], type="vote", OOB=T) 
}

votes.all <- Reduce('+', votes)

predictions.rf <- c()
for (i in 1:nrow(votes.all)){
  if (votes.all[i,1] > votes.all[i,2]) {
    predictions.rf[i] <- 0 # Deceased
  } else {
    predictions.rf[i] <- 1 # Survived
  }
}

table(predictions.rf) # 0 = deceased, 1 = survived

# Prepare final submission
write.csv(data.frame(PassengerID = 892:1309, Survived = predictions.rf), 
          file = "MyPredictionsRF", quote=F, row.names=F)

################################### GRADIENT BOOSTING #####################################

predictions.gbm <- matrix(list(), nrow = 20, ncol = 20)
for (m_1 in 1:20){
  
  gbm.model <- gbm(as.numeric(Survived)-1 ~ ., data = imputeddat[[m_1]],
                   distribution = "bernoulli",
                   n.trees = 200,
                   interaction.depth = 4,
                   train.fraction = 0.9,
                   shrinkage = 0.01)
  
  for (m_2 in 1:20){
    
    predictions.gbm[m_1, m_2] <- data.frame(
      predict.gbm(gbm.model,
                  newdata = imputedtestdat[[m_2]],
                  n.trees = 200,
                  type = "response"))
  }
}

votes.all.gmb <- ifelse(Reduce('+', predictions.gbm)/400 > 0.5, 1, 0)

table(votes.all.gmb)

# Prepare final submission
write.csv(data.frame(PassengerID = 892:1309, Survived = votes.all.gmb), 
          file = "MyPredictionsGBM", quote=F, row.names=F)