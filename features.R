library(tidyverse)
library(here)
library(reticulate)
library(doParallel)

## Cluster
nCores<- detectCores()-2
registerDoParallel(makeCluster(nCores))

## Data
train_x<- readRDS(here("data", "train_x.rds"))
valid_x<- readRDS(here("data", "valid_x.rds"))
test_x<- readRDS(here("data", "test_x.rds"))

## Averages the three colors
colorMean<- function(df){
  dims<- dim(df)
  cols<- dims[2]/3
  feats<- NULL
  for(i in 1:(dims[1])){
    ms<- c()
    if(i%%100 == 0){print(paste("curently at row:", i))}
    for(j in 1:cols){
      ms<- c(ms, mean(df[c(j, j+cols, j+2*cols)]))
    }
    feats<- rbind(feats, ms)
  }
  feats
}

## searches for edges defined by large change in level of pixel moving left to right
## Slow don't use on large datasets
edge<- function(df, dims){
  res<- NULL
  for(i in 1:nrow(df)) {
    ar<- array(df[i,], dim = dims)
    edgs<- array(0, dim = dims)
    if(i%%100 == 0){print(paste("Observation: ",i))}
    for(j in 2:dims[2]){
      edgs[,j]<-as.numeric(abs(ar[,j]-ar[,(j-1)])>=0.2)
    }
    res<- rbind(res, array(edgs))
  }
  res
}

## second attempt. Much faster, but loses a column of data. Fine for xgboost but not keras
edge2<- function(df, dims){
  res<- NULL
  cvar<- dims[2]
  res<- foreach(i = 1:nrow(df), .combine = rbind) %dopar% {
    ar<- array(df[i,], dim = dims)
    edgs<- array(ar[,1:(cvar-1)] - ar[,2:cvar])
#    if(i%%100 == 0){print(paste("Observation: ",i))}
  }
  return(res)
}

## zero column back in to allow forming of 3d array
edge3<- function(df, dims){
  res<- NULL
  cvar<- dims[2]
  res<- foreach(i = 1:nrow(df), .combine = rbind) %dopar% {
    ar<- array(df[i,], dim = dims)
    edgs<- array(cbind((ar[,1:(cvar-1)] - ar[,2:cvar]), rep(0, dims[1]))) 
    #    if(i%%100 == 0){print(paste("Observation: ",i))}
  }
  return(res)
}

## vert and horizontal edge
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


## Average all three colors
#mu.t<-colorMean(train_x)
#mu.v<-colorMean(valid_x)

## find edges in avaerage data
#dims<- c(32,32)
#ed.t<- edge2(mu.t, dims)
#ed.v<- edge2(mu.v, dims)

## save df with average and average edge
#train_x2<- cbind(train_x, mu.t, ed.t)
#valid_x2<- cbind(valid_x, u, ed)

#saveRDS(train_x2, "train_x2.rds")
#saveRDS(valid_x2, "valid_x2.rds")
## averaging the colors for each pixel and finding the edges of the averaged
## values did not add any value to model fitting. 

## find edges within each color
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

train_x3<- cbind(train_x, ed1.t, ed2.t, ed3.t)
valid_x3<- cbind(valid_x, ed1.v, ed2.v, ed3.v)
test_x3<- cbind(test_x, ed1.test, ed2.test, ed3.test)

saveRDS(train_x3, "data/train_x3.rds")
saveRDS(valid_x3, "data/valid_x3.rds")
saveRDS(test_x3, "data/test_x3.rds")
## edge features based on each of the three colors. This increased the validation
## accuracy of the xgboost model. 

gc()
