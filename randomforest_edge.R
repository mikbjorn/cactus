## libraries
library(tidyverse)
library(here)
library(ranger)
## Data
train_x<- readRDS(here("data", "train_x3.rds"))
train_y<- readRDS(here("data", "train_y.rds"))
valid_x<- readRDS(here("data", "valid_x3.rds"))
valid_y<- readRDS(here("data", "valid_y.rds"))
test_x<- readRDS(here("data", "test_x3.rds"))
test_y<- readRDS(here("data", "test_y.rds"))

#view(train_x)
## Random Forest - slow, DO NOT RUN

#md1<- randomForest(train_x, as.factor(train_y),
#                   xtest = valid_x, ytest = as.factor(valid_y),
#                   importance = T, proximity = T)
#md1Pred = (md1$test)$predicted
#
#table(validy, md1Pred)
#
#varImpPlot(md1)

## Random Forest Fast
df.train<- cbind(train_x,y = as.factor(train_y+1))%>% data.frame()
df.valid<- cbind(valid_x, y = as.factor(valid_y+1))%>% data.frame()
df.test<- cbind(test_x, y = as.factor(test_y+1))%>% data.frame()
## sample size for rf
mt = round(dim(train_x)[2]^0.5)

## fit many models of different depths - 
depth = c(1,5,10,50,100,150,200,500,750,1000)
#depth <- seq(10,100,5)
v.acc<- c()
t.acc<- c()
for(i in 1:length(depth)){
  assign(paste('rf', depth[i], sep = ''), 
         ranger(y~., data = df.train, num.trees = depth[i], mtry = mt, importance = 'impurity', oob.error = T,
                                                 verbose=T, classification = T))
  p<- predict(get(paste('rf', depth[i],sep = '')), data=df.valid)
  v.acc<- c(v.acc, mean(p$predictions == df.valid$y))
  pt<- get(paste('rf', depth[i],sep = ''))$predictions
  pt[is.na(pt)]<- NA
  t.acc<- c(t.acc, mean(pt == df.train$y, na.rm=T))
}

acc<- data.frame(depth,v.acc,t.acc)

p1<-ggplot(acc, aes(x=depth))+
  geom_point(aes(y=1-v.acc))+
  geom_line(aes(y=1-v.acc))+
  geom_point(aes(y=1-t.acc), color='blue')+
  geom_line(aes(y=1-t.acc), color='blue')+
  scale_x_continuous(breaks = seq(0,1000,100))


## fit many models of different mtrys
mtrys = seq(1, 401, 20)
v.acc<- c()
t.acc<- c()
for(i in 1:length(mtrys)){
  assign(paste('rf', mtrys[i], sep = ''), 
         ranger(y~., data = df.train, num.trees = 150, mtry = mtrys[i], importance = 'impurity', oob.error = T,
                verbose=T, classification = T))
  p<- predict(get(paste('rf', mtrys[i],sep = '')), data=df.valid)
  v.acc<- c(v.acc, mean(p$predictions == df.valid$y))
  pt<- get(paste('rf', mtrys[i],sep = ''))$predictions
  pt[is.na(pt)]<- NA
  t.acc<- c(t.acc, mean(pt == df.train$y, na.rm=T))
}

acc<- data.frame(mtrys,v.acc,t.acc)

ggplot(acc, aes(x=mtrys))+
  geom_point(aes(y=1-v.acc))+
  geom_smooth(aes(y=1-v.acc))+
  geom_point(aes(y=1-t.acc), color='blue')+
  geom_smooth(aes(y=1-t.acc), color='blue')

acc[which.max(v.acc),]


## final model
md<- ranger(y~., data = df.train, num.trees = 150, mtry = 301, 
            importance = 'impurity', oob.error = T, verbose=T, probability = T)

## validation accuracy
rf.pred<- predict(md, df.valid)
table(rf.pred$predictions[,1]>0.5 , valid_y)

mean(as.numeric(rf.pred$predictions[,1]>0.5)==valid_y)

## training accuracy
rf.tr.pred<- predict(md, df.train)
mean(as.numeric(rf.tr.pred$predictions[,1]>0.5)==train_y)

## test accuracy
rf.test.pred<- predict(md, df.test)
mean(as.numeric(rf.test.pred$predictions[,1]>0.5)==test_y)

saveRDS(rf.pred$predictions[,1], "rf_vpred.RDS")
saveRDS(rf.tr.pred$predictions[,1], "rf_trpred.RDS")
saveRDS(rf.test.pred$predictions[,1], "rf_tepred.RDS")
