library(tidyverse)
library(here)
library(xgboost)
library(colorspace)

pal<-hcl.colors(7)

## Data
train_x<- readRDS(here("data", "train_x3.rds"))
train_y<- readRDS(here("data", "train_y.rds"))
valid_x<- readRDS(here("data", "valid_x3.rds"))
valid_y<- readRDS(here("data", "valid_y.rds"))
test_x<-  readRDS(here("data", "test_x3.rds"))

dtrain<- xgb.DMatrix(data = train_x, label=train_y)
dvalid<- xgb.DMatrix(data = valid_x, label=valid_y)

watchlist<- list(train=dtrain, test=dvalid)

## Train tree model - This tree based model works best
## depth = 4
md1<- xgb.train(data = dtrain, max.depth=4, nthread=7, nrounds = 2000,
               eta=0.2, watchlist = watchlist, early_stopping_rounds = 15,
             objective = "binary:logistic", verbose = 1, eval_metric = "error")


md1$evaluation_log[which.min(unlist(md1$evaluation_log[,3]))[[1]]]


## depth = 5
md2<- xgb.train(data = dtrain, max.depth=5, nthread=7, nrounds = 2000,
                eta=0.2, watchlist = watchlist, early_stopping_rounds = 15,
                objective = "binary:logistic", verbose = 1, eval_metric = "error")


md2$evaluation_log[which.min(unlist(md2$evaluation_log[,3]))[[1]]]

md1$evaluation_log %>% tibble()%>%
  ggplot(aes(x=iter, y=train_error))+
  geom_point(col=pal[1])+
  geom_line(col=pal[1])+
  geom_point(aes(y=test_error), col=pal[2])+
  geom_line(aes(y=test_error), col=pal[2])+
  geom_vline(xintercept = 182, col=pal[3])+
  theme_classic()+
  labs(x="Iteration", y="Error", title = "XGBoost Model")


## depth = 6
md3<- xgb.train(data = dtrain, max.depth=7, nthread=6, nrounds = 2000,
                eta=0.2, watchlist = watchlist, early_stopping_rounds = 15,
                objective = "binary:logistic", verbose = 1, eval_metric = "error")


md3$evaluation_log[which.min(unlist(md3$evaluation_log[,3]))[[1]]]

## depth = 3
md4<- xgb.train(data = dtrain, max.depth=3, nthread=7, nrounds = 2000,
                eta=0.2, watchlist = watchlist, early_stopping_rounds = 15,
                objective = "binary:logistic", verbose = 1, eval_metric = "error")


md4$evaluation_log[which.min(unlist(md4$evaluation_log[,3]))[[1]]]


## Train linear model - linear model does not perform well
md5<- xgb.train(data = dtrain, booster="gblinear", max.depth=3, nthread=6, nrounds = 200, watchlist = watchlist,
                objective = "binary:logistic", verbose = 1, eval_metric = "error", early_stopping_rounds = 10)


md5$evaluation_log[which.min(unlist(md5$evaluation_log[,3]))[[1]]]


## model 2 worked best
md2.tr.pred<- predict(md2, train_x)
md2.va.pred<- predict(md2, valid_x)


## Test data
test_x<- readRDS(here("data", "test_x3.rds"))
test_y<- readRDS(here("data", "test_y.rds"))

md2.te.pred<- predict(md2, test_x)
pred<- as.numeric(md2.te.pred>0.5)

mean(pred==test_y)
## achieves 0.044 error

saveRDS(md2.tr.pred, "xgb_trpred.RDS")
saveRDS(md2.va.pred, "xgb_vpred.RDS")
saveRDS(md2.te.pred, "xgb_tepred.RDS")
