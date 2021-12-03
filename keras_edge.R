#library(tidyverse)
library(EBImage)
library(keras)
library(tensorflow)
library(tidymodels)
library(here)
#library(reticulate)

#conda_python(envname = "TF2R")

## Data
train_x<- readRDS(here("data", "train_x3.rds"))
train_y<- readRDS(here("data", "train_y.rds"))
valid_x<- readRDS(here("data", "valid_x3.rds"))
valid_y<- readRDS(here("data", "valid_y.rds"))
test_x<- readRDS(here("data", "test_x3.rds"))
test_y<- readRDS(here("data", "test_y.rds"))

train_x_reshape = array_reshape(x = train_x, dim = list(10500, 32, 32, 9))
valid_x_reshape = array_reshape(x = valid_x, dim = list(3500, 32, 32, 9))
test_x_reshape = array_reshape(x = test_x, dim = list(3500, 32, 32, 9))


## Response variable (hot coding)
trainLabels= to_categorical(train_y)
validLabels= to_categorical(valid_y)

# Create Model
model <- keras_model_sequential()

model %>% 
  layer_conv_2d(filters = 32, kernel_size = c(3,3), activation = "relu", 
                input_shape = c(32,32,9)) %>% 
  layer_max_pooling_2d(pool_size = c(2,2)) %>% 
  layer_conv_2d(filters = 64, kernel_size = c(3,3), activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = c(2,2)) %>% 
  layer_conv_2d(filters = 64, kernel_size = c(3,3), activation = "relu")

model %>% 
  layer_flatten() %>% 
  layer_dense(units = 64, activation = "relu") %>% 
  layer_dense(units = 2, activation = "softmax")  

summary(model)

## Compile Model
model %>%
  compile(loss = 'binary_crossentropy',
          optimizer = optimizer_adam(),
          #optimizer = optimizer_rmsprop(lr = 0.0001, decay = 1e-6),
          metrics = 'accuracy')

## Fit the model
fitModel <- model %>%
  fit(train_x_reshape,
      trainLabels,
      epochs = 50, # one pass over the entire dataset, evalutate at end of each epoch
      batch_size = 500, # number of samples processed independently, in parallel. 1 update/batch 
      validation_data = list(valid_x_reshape,validLabels)
  )

## Plot the model fitting
plot(fitModel)

# Evaluation & Prediction

## Evaluate Training data (accuracy / loss)
model %>% evaluate(train_x_reshape, trainLabels)

## Evaluate Validation Data (accuracy / loss)
model %>% evaluate(valid_x_reshape,validLabels)

## Results (table)

pred.prob <- model %>% predict(valid_x_reshape)
pred<- as.numeric(pred.prob[,2]>0.7 )

table(Predicted = pred, Actual = valid_y)
mean(pred == valid_y)


# Classification: Predicted vs. Actual 

## You can determine which images were misclassified & probability of classifications
prob <- model %>% predict(valid_x_reshape)
cbind(prob, Predicted = pred, Actual = valid_y)

## Results test

pred.prob.test <- model %>% predict(test_x_reshape)
pred.test<- as.numeric(pred.prob.test[,2]>0.7 )
mean(pred.test == test_y)

pred.prob.tr <- model %>% predict(train_x_reshape)


saveRDS(pred.prob[,2], "keras_vpred.RDS")
saveRDS(pred.prob.test[,2], "keras_tepred.RDS")
saveRDS(pred.prob.tr[,2], "keras_trpred.RDS")

