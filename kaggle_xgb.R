## libraries
library(tidyverse)
library(caret)
library(here)
library(EBImage)
library(doParallel)
library(reticulate)
library(xgboost)

## data
train_path <- './train'
test_path <- './test'

#train_path <- here("data","train")
#test_path <- here("data", "test")

## Cluster
nCores<- detectCores()-1
registerDoParallel(makeCluster(nCores))

## Image processing
proc_img<- function(file){
  image<- readImage(file)
  #image.cl<- clahe(image)
  #image.n<- normalize(image.cl)
  array_reshape(image,c(32,32,3))
}

edge3<- function(df, dims){
  res1<- NULL
  cvar<- dims[2]
  res1<- foreach(i = 1:nrow(df), .combine = rbind) %dopar% {
    ar<- array(df[i,], dim = dims)
    edgs<- array(cbind((ar[,1:(cvar-1)] - ar[,2:cvar]), rep(0, dims[1]))) 
    #    if(i%%100 == 0){print(paste("Observation: ",i))}
  }
  res2<- NULL
  cvar<- dims[2]
  res2<- foreach(i = 1:nrow(df), .combine = rbind) %dopar% {
    ar<- array(df[i,], dim = dims)
    edgs<- array(rbind((ar[1:(cvar-1),] - ar[2:cvar,]), rep(0, dims[2]))) 
    #    if(i%%100 == 0){print(paste("Observation: ",i))}
  }
  res<- cbind(res1,res2)
  return(res)
}


## load training data
train_csv <- read.csv("../input/aerial-cactus-identification/train.csv")
test_labels<- data.frame(id = list.files(test_path))

#train_csv <- read.csv(here("data", "train.csv"))
#test_labels<- data.frame(id = list.files(here("data", "test")))

## training split
training<- createDataPartition(train_csv$has_cactus, p=0.8, list = F)

train_imgs<- paste(train_path, train_csv$id[training], sep = '/')
valid_imgs<- paste(train_path, train_csv$id[-training], sep = '/')
test_imgs<- paste(test_path, test_labels$id, sep = '/')

## answers split
train_y<- train_csv$has_cactus[training]
valid_y<- train_csv$has_cactus[-training]

## Load training images

dims <- length(train_imgs)
train_x = NULL
train_x<- foreach(i = 1:dims, .packages = c("EBImage","reticulate"), .combine = rbind) %dopar% {
  train_x[[i]] <- proc_img(train_imgs[i])
}

dims <- length(valid_imgs)
valid_x = NULL
valid_x<- foreach(i = 1:dims, .packages = c("EBImage","reticulate"), .combine = rbind) %dopar% {
  valid_x[[i]] <- proc_img(valid_imgs[i])
}

## test images
dims <- length(test_imgs)
test_x = NULL
test_x<- foreach(i = 1:dims, .packages = c("EBImage","reticulate"), .combine = rbind) %dopar% {
  test_x[[i]] <- proc_img(test_imgs[i])
}


## Edge features
dims<- c(32,32)
ed1.t<- edge3(train_x[,1:1024], dims)
ed2.t<- edge3(train_x[,1025:2048], dims)
ed3.t<- edge3(train_x[,2049:3072], dims)

ed1.v<- edge3(valid_x[,1:1024], dims)
ed2.v<- edge3(valid_x[,1025:2048], dims)
ed3.v<- edge3(valid_x[,2049:3072], dims)

ed1.test<- edge3(test_x[,1:1024], dims)
ed2.test<- edge3(test_x[,1025:2048], dims)
ed3.test<- edge3(test_x[,2049:3072], dims)

train_x<- cbind(train_x, ed1.t, ed2.t, ed3.t)
valid_x3<- cbind(valid_x, ed1.v, ed2.v, ed3.v)
test_x<- cbind(test_x, ed1.test, ed2.test, ed3.test)

## xgboost
dtrain<- xgb.DMatrix(data = train_x, label=train_y)

watchlist<- list(train=dtrain, test=dtest)

md.xgb<- xgb.train(data = dtrain, max.depth=4, nthread=7, nrounds = 2000,
                eta=0.2,  watchlist = watchlist, early_stopping_rounds = 15,
                objective = "binary:logistic", verbose = 1, 
                eval_metric = "error")

preds<- predict(md.xgb, test_x) 
preds<- as.numeric(preds>0.5)
submission<- data.frame(id=test_labels$id , has_cactus=preds)
str(submission)

write_csv(submission, "submission.csv")
