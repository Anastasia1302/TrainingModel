Model Training . ML Course Project
Author: Anastasia


The goal of this assignment is prediction of the manner in which people do their exercise. We have training data set to train the model and testing data set to predict 20 different manners of doing the excersize. "Classe" variable in the training data set is the variable that we want to predict. It shows how people do thier excersizes. Any other varibal in the training data set can be used as a predictor. 

In the following report I describe how the prediction model is built, I discuss its accuracy and I show the results of my predictions of 20 different test cases form testing data set. 

First, I save and read training and testing data from csv files:

``` 
setwd("~/Dropbox/Coursera/ML/project")

library(data.table)

DataTrain= read.csv("pml-training.csv", header=TRUE, sep=",")
DataTest = read.csv("pml-testing.csv", header=TRUE, sep=",")

DWork<- data.table(DataTrain)
DTest<- data.table (DataTest)

```

Then, I make my first selection of predictors using those that have smaller number of NAs. More specifically, I select as predictors only those variables that do not have missing observations in the test dataset. These variables are arm, dumbbell, and forearm. I update inirial datasets with these new predictor candidates. 

```
# Find columns with any missings

AnyMissCols <- sapply(DWork, function(x)any(is.na(x) | x == ""));  
print(AnyMissCols)
ColsAnyMiss <-names(AnyMissCols[AnyMissCols>0]);    
print("Columns with some values missing");    
print(ColsAnyMiss)

# Select Candidates

isPredictor <- !AnyMissCols & grepl("belt|arm|dumbbell|forearm", names(AnyMissCols))
Candidates <- names(AnyMissCols)[isPredictor]

# Update dataset with new Candidates - predictors 
Include <- c("classe", Candidates)
Include
DWork <- DWork[, Include, with=FALSE]

```

I then proceed to splitting the training data into training and testing partitions. Before that, I check whether "classe" variable is factor object:
```
# check levels

levels(DWork$classe) # ok.
summary(DWork$classe)

# split data into 60% trainig and %40 testing

library(caret); set.seed(333);
inTrain<- createDataPartition(y=DWork$classe,
                              p=0.6)

train<- DWork[inTrain[[1]]] 
test<- DWork[-inTrain[[1]]]

dim(train)
dim(test)
```
I standardize prediction variables in train and test partitions and I check for prediction variables with zero variation. Ther are no variables with zero variability among selected predictors. 

```
# Train partition

Xme <- train[, Candidates, with=FALSE]
preObj<- preProcess(Xme, method=c("center","scale"))
preObj
XCSme <- predict(preObj, Xme)
trainCS<- data.table(data.frame(classe = train[, classe], XCSme))

# Test partition

Yme <- test[, Candidates, with=FALSE]
YCSme <- predict(preObj, Yme)
testCS<- data.table(data.frame(classe = test[,classe], YCSme))


# Check for variables with no variation

nzv <- nearZeroVar(trainCS, saveMetrics=TRUE)
if (any(nzv$nzv)) nzv else message("No variables with near zero variance")

```

I set parallel clusters and I proceed to the model training by Random Forest Method. I use this methode becuase it yields small out of smapel error.

```
# set parallel

require(parallel)
require(doParallel)
cl <- makeCluster(detectCores() - 1)
registerDoParallel(cl)

getDoParWorkers() 

# set control parameters
ctrl <- trainControl(classProbs=TRUE,
                     savePredictions=TRUE,
                     allowParallel=TRUE)

# train the model
method <- "rf"
trainingModel <- train(classe ~ ., data=trainCS, method=method, trControl=ctrl)

stopCluster(cl)
```

I evaluate model on the training and testing datasets
```
# evaluate model on the train dataset

trainingModel

pred<- predict(trainingModel, trainCS)
confusionMatrix(pred, train[,classe])

# evaluate model on the test dataset

hat <- predict(trainingModel, testCS)
confusionMatrix(hat, testCS[, classe])
```

I display and save the final model
```
# display the final model

varImp(trainingModel)
trainingModel$finalModel # the estimated error rate is 0.83%.


# save model 
save(trainingModel, file="trainingModel.RData")
```


Finally, I predict on the training data

```
load(file="trainingModel.RData", verbose=TRUE)

DTestCS <- predict(preObj, DTest[, Candidates, with=FALSE])
hat <- predict(trainingModel, DTestCS)
DTest <- cbind(hat , DTest)
subset(DTest, select=names(DTest)[grep("belt|arm|dumbbell|forearm", names(DTest), invert=TRUE)])

```
I use the following code to generate txt files with the results for submission to Coursera:

```
pml_write_files = function(x){
    n = length(x)
    for(i in 1:n){
        filename = paste0("problem_id_",i,".txt")
        write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
    }
}

setwd("~/Dropbox/Coursera/ML/project/answers")

pml_write_files(hat)

```
