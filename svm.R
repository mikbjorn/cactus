# Libraries
library(tidyverse)
library(reticulate)
library(parallelSVM)
library(here)
library(e1071)

# SVM
#Importing the data created by Project 2 Parallel Coding.R

trainx <- readRDS("data/train_x.rds")
trainy <- readRDS("data/train_y.rds")

validx <- readRDS("data/valid_x.rds")
validy <- readRDS("data/valid_y.rds")

testx <- readRDS("data/test_x.rds")
testy <- readRDS("data/test_y.rds")

train_x_reshape = array_reshape(x = trainx, dim = list(10500, 32, 32, 3))
valid_x_reshape = array_reshape(x = validx, dim = list(3500, 32, 32, 3))
test_x_reshape = array_reshape(x = testx, dim = list(3500, 32, 32, 3))


#transforming the data
df_train <- data.frame(train_x_reshape)
fact_trainy <- as.factor(trainy)

df_validx <- data.frame(valid_x_reshape)
fact_validy <- as.factor(validy)

df_testx <- data.frame(test_x_reshape)
fact_testy <- as.factor(testy)

#model builing and validation
svm.mod <- parallelSVM(x = df_train,y = fact_trainy, kernel = 'radial', type = "C-classification")
predictions <- predict(svm.mod, df_validx)
acc_tab <- table(predictions, fact_validy)
acc_tab
accuracy <- (acc_tab[1]+acc_tab[4])/sum(acc_tab)
accuracy

pred.train<- predict(svm.mod, df_train)
pred.acc<- mean(pred.train == fact_trainy)
table(pred.train, fact_trainy)

#test predictions
test_pred <- predict(svm.mod, df_testx)
test_tab <- table(predictions, fact_testy)
test_tab
test_accuracy <- (test_tab[1]+test_tab[4])/sum(test_tab)
test_accuracy