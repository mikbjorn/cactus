library(partitions)
library(tidyverse)
library(here)

## estimates
## horizontal
cnn_te<- readRDS(here("est", "cnn_te.RDS"))
cnn_tr<- readRDS(here("est", "cnn_tr.RDS"))
cnn_va<- readRDS(here("est", "cnn_va.RDS"))
rf_te<- readRDS(here("est", "rf.test.RDS"))
rf_tr<- readRDS(here("est", "rf.train.RDS"))
rf_va<- readRDS(here("est", "rf.valid.RDS"))
xgb_te<- readRDS(here("est", "xgb_te.RDS"))
xgb_tr<- readRDS(here("est", "xgb_tr.RDS"))
xgb_va<- readRDS(here("est", "xgb_va.RDS"))

h_est_te<- data.frame(k=cnn_te, x=xgb_te, r=rf_te)%>% as.matrix()
h_est_tr<- data.frame(k=cnn_tr, x=xgb_tr, r=rf_tr)%>% as.matrix()
h_est_va<- data.frame(k=cnn_va, x=xgb_va, r=rf_va)%>% as.matrix()

## hor and vert
xgb.va<- readRDS("xgb_vpred.RDS")
rf.va<- readRDS("rf_vpred.RDS")
ker.va<- readRDS("keras_vpred.RDS")
xgb.tr<- readRDS("xgb_trpred.RDS")
rf.tr<- readRDS("rf_trpred.RDS")
ker.tr<- readRDS("keras_trpred.RDS")
xgb.te<- readRDS("xgb_tepred.RDS")
rf.te<- readRDS("rf_tepred.RDS")
ker.te<- readRDS("keras_tepred.RDS")

hv_est_te<- data.frame(k=ker.te, x=xgb.te, r=rf.te)%>% as.matrix()
hv_est_tr<- data.frame(k=ker.tr, x=xgb.tr, r=rf.tr)%>% as.matrix()
hv_est_va<- data.frame(k=ker.va, x=xgb.va, r=rf.va)%>% as.matrix()

## truth
valid_y<- readRDS(here("data", "valid_y.rds"))
train_y<- readRDS(here("data", "train_y.rds"))
test_y<- readRDS(here("data", "test_y.rds"))


##Averaging function
average_train<- function(est, y){
  numparts<- ncol(est)
  sumparts<- 500
  weights<- compositions(n=sumparts, m=numparts, include.zero=TRUE)/sumparts 
  
  acc<- c()
  for (i in 1:ncol(weights)){
    preds<- rowSums(t(t(est)*weights[,i]))
    acc<- c(acc, mean(as.numeric(preds>0.5)== y))
    }
  cbind(t(weights[,which.max(acc)]), max(acc))
}

average<- function(result, model_weights){
  res<- rowSums(t(t(result)* model_weights))
  data.frame(prob=res, class = as.numeric(res>0.5))
}

## training
md<- average_train(h_est_va, valid_y) ## k=0.734, xgb=0.266, rf=0
h_est_train<- average(h_est_tr, md[,1:3])
mean(h_est_train$class == train_y)
h_est_valid<- average(h_est_va, md[,1:3])
mean(h_est_valid$class == valid_y)
h_est_test<- average(h_est_te, md[,1:3])
mean(h_est_test$class == test_y)

md2<- average_train(hv_est_va, valid_y) ## k=0.478, xgb=0.488, rf=0.034
hv_est_train<- average(hv_est_tr, md[,1:3])
mean(hv_est_train$class == train_y)
hv_est_valid<- average(hv_est_va, md[,1:3])
mean(hv_est_valid$class == valid_y)
hv_est_test<- average(hv_est_te, md[,1:3])
mean(hv_est_test$class == test_y)





mean(as.numeric(xgb>0.5)==valid_y)
mean(as.numeric(rf>0.5)==valid_y)
mean(as.numeric(ker>0.7)==valid_y)
