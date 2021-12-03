library(tidyverse)
library(caret)
library(here)
library(EBImage)
library(doParallel)
library(reticulate)

set.seed(496)
## Cluster
nCores<- detectCores()-1
registerDoParallel(makeCluster(nCores))

## Image processing
proc_img<- function(file){
  image<- readImage(file)
  image<- clahe(image)
  array_reshape(image,c(32,32,3))
}

## load training data
train_csv <- read.csv(here("data", "train.csv"))

## Image paths
train_img_path <- here("data", "train") # path of Kaggle training images
test_image_path <- here("data", "test")


## training split
training<- createDataPartition(train_csv$has_cactus, p=0.6, list = F)

train_imgs<- paste(train_img_path, train_csv$id[training], sep = '/')
valid_imgs<- paste(train_img_path, train_csv$id[-training], sep = '/')
train_y<- train_csv$has_cactus[training]
valid_y<- train_csv$has_cactus[-training]

## Testing Split
testing<- createDataPartition(valid_y, p=0.5, list = F )

test_imgs<- valid_imgs[testing]
valid_imgs<- valid_imgs[-testing]
test_y<- valid_y[testing]
valid_y<- valid_y[-testing]

## Load images

dims <- length(train_imgs)
train_x <- NULL
train_x<- foreach(i = 1:dims, .packages = c("EBImage","reticulate"), .combine = rbind) %dopar% {
  train_x[[i]] <- proc_img(train_imgs[i])
}

dim(train_imgs)

dims <- length(valid_imgs)
valid_x <- NULL
valid_x<- foreach(i = 1:dims, .packages = c("EBImage","reticulate"), .combine = rbind) %dopar% {
  valid_x[[i]] <- proc_img(valid_imgs[i])
}

dim(valid_imgs)

dims <- length(test_imgs)
test_x <- NULL
test_x<- foreach(i = 1:dims, .packages = c("EBImage","reticulate"), .combine = rbind) %dopar% {
  test_x[[i]] <- proc_img(test_imgs[i])
}

dim(test_imgs)

## garbage control
invisible(gc())
stopImplicitCluster()

saveRDS(train_x, "train_x2.rds")
saveRDS(train_y, "train_y2.rds")
saveRDS(valid_x, "valid_x2.rds")
saveRDS(valid_y, "valid_y2.rds")
saveRDS(test_x, "test_x2.rds")
saveRDS(test_y, "test_y2.rds")








