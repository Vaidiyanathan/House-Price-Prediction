
# Individual Case 1 -------------------------------------------------------

# load Boston data
library(MASS)
data(Boston)
set.seed(12969599)
index <- sample(nrow(Boston),nrow(Boston)*0.75)
sample_index <- index
boston.train <- Boston[index,]
boston.test <- Boston[-index,]

Boston.X.std <- scale(dplyr::select(Boston, -medv))

X.train<- as.matrix(Boston.X.std)[sample_index,]
X.test<-  as.matrix(Boston.X.std)[-sample_index,]
Y.train<- Boston[sample_index, "medv"]
Y.test<- Boston[-sample_index, "medv"]
Boston_train <- Boston[sample_index,]
Boston_test <- Boston[-sample_index,]


# Linear regression -------------------------------------------------------

model_1 <- lm(medv~., data=boston.train)
summary(model_1)
#in sample MSE
summary(model_1)$sigma^2
# 20.84358

# Out of Sample MSE
boston.test_lr <- predict( model_1, newdata = boston.test)
mean((boston.test$medv-boston.test_lr)^2)
# 28.23323

# Stepwise ----------------------------------------------------------------

library(leaps)
#Stepwise, Backward and Forward selection:
nullmodel=lm(medv~1, data=boston.train)
fullmodel=lm(medv~., data=boston.train)

#Stepwise Selection:
model_step_s <- step(nullmodel, scope=list(lower=nullmodel, upper=fullmodel),
                     direction='both')
summary(model_step_s)
summary(model_step_s)$sigma^2
# 20.77397

# Out of Sample MSE
boston.test_lr <- predict( model_step_s, newdata = boston.test)
mean((boston.test$medv-boston.test_lr)^2)
# 28.05547


# Decision Tree -----------------------------------------------------------

library(rpart)
library(rpart.plot)

set.seed(12969599)

boston.rpart <- rpart(formula = medv ~ ., data = boston.train)

boston.train.pred.tree = predict(boston.rpart)
mean((Y.train - boston.train.pred.tree)^2)
# 15.05392

# Out of sample
boston.test.pred.tree = predict(boston.rpart,Boston_test)
mean((Y.test - boston.test.pred.tree)^2)
# 25.01981

# Bagging -----------------------------------------------------------------

library(ipred)

boston.bag<- bagging(medv~., data = boston.train, nbagg=100)
boston.bag
boston.bag.pred<- predict(boston.bag, newdata = boston.train)
mean((boston.train$medv-boston.bag.pred)^2)
# 11.51375

# How many trees are goood

ntree<- c(1, 3, 5, seq(10, 200, 10))
MSE.test<- rep(0, length(ntree))
for(i in 1:length(ntree)){
  boston.bag1<- bagging(medv~., data = boston.train, nbagg=ntree[i])
  boston.bag.pred1<- predict(boston.bag1, newdata = boston.test)
  MSE.test[i]<- mean((boston.test$medv-boston.bag.pred1)^2)
}
plot(ntree, MSE.test, type = 'l', col=2, lwd=2, xaxt="n")
axis(1, at = ntree, las=1)

# number of trees that result in lowest test MSE
ntree[which(MSE.test==min(MSE.test))]
# 40

# lowest test MSE
MSE.test[MSE.test==min(MSE.test)]
# [1] 14.75407

# Out of Bag

boston.bag.oob<- bagging(medv~., data = boston.train, coob=T, nbagg=100)
boston.bag.oob$err^2


# Random Forest -----------------------------------------------------------

library(randomForest)
boston.rf<- randomForest(medv~., data = boston.train, importance=TRUE)
boston.rf
boston.rf$importance

plot(boston.rf$mse, type='l', col=2, lwd=2, xlab = "ntree", ylab = "OOB Error")

#prediction
boston.rf.pred<- predict(boston.rf)
mean((boston.train$medv-boston.rf.pred)^2)
# 10.6484

oob.err<- rep(0, 13)
test.err<- rep(0, 13)
for(i in 1:13){
  fit<- randomForest(medv~., data = boston.train, mtry=i)
  oob.err[i]<- fit$mse[500]
  test.err[i]<- mean((boston.test$medv-predict(fit, boston.test))^2)
  cat(i, " ")
}

matplot(cbind(test.err, oob.err), pch=15, col = c("red", "blue"), type = "b",
        ylab = "MSE", xlab = "mtry")
legend("topright", legend = c("test Error", "OOB Error"), pch = 15,
       col = c("red", "blue"))


# number of predictors resulting in lowest test mse
which(test.err==min(test.err))
# 11

# lowest test mse
test.err[which(test.err==min(test.err))]
# 10.24894


# Boosting ----------------------------------------------------------------

library(gbm)
boston.boost<- gbm(medv~., data = boston.train, distribution = "gaussian",
                   n.trees = 10000, shrinkage = 0.01, interaction.depth = 8)
summary(boston.boost)

par(mfrow=c(2,2))
plot(boston.boost, i="lstat")
plot(boston.boost, i="rm")

#prediction
boston.boost.pred.test<- predict(boston.boost, n.trees = 10000)
mean((boston.train$medv-boston.boost.pred.test)^2)
# 0.01857575


#prediction
boston.boost.pred.test<- predict(boston.boost, newdata = boston.test, n.trees = 10000)
mean((boston.test$medv-boston.boost.pred.test)^2)
# 10.81737

par(mfrow = c(1,1))
ntree<- seq(100, 10000, 100)
predmat<- predict(boston.boost, newdata = boston.test, n.trees = ntree)
err<- apply((predmat-boston.test$medv)^2, 2, mean)
plot(ntree, err, type = 'l', col=2, lwd=2, xlab = "n.trees", ylab = "Test MSE")
abline(h=min(test.err), lty=2)




# Neural Net --------------------------------------------------------------


library(MASS)
maxs <- apply(Boston, 2, max) 
mins <- apply(Boston, 2, min)

set.seed(12969599)
scaled <- as.data.frame(scale(Boston, center = mins, scale = maxs - mins))
index <- sample(1:nrow(Boston),round(0.75*nrow(Boston)))

train_ <- scaled[index,]
test_ <- scaled[-index,]

library(neuralnet)
## Warning: package 'neuralnet' was built under R version 3.5.2
n <- names(train_)
f <- as.formula(paste("medv ~", paste(n[!n %in% "medv"], collapse = " + ")))
nn <- neuralnet(f,data=train_,hidden=c(5,3),linear.output=T)
plot(nn)
pr.nn <- neuralnet::compute(nn,test_[,1:13])

pr.nn_ <- pr.nn$net.result*(max(Boston$medv)-min(Boston$medv))+min(Boston$medv)
test.r <- (test_$medv)*(max(Boston$medv)-min(Boston$medv))+min(Boston$medv)
train.predict <- predict(nn, train_)
mean((train.predict - train_$medv)^2 )

# MSE of testing set
MSE.nn <- sum((test.r - pr.nn_)^2)/nrow(test_)
MSE.nn

# GAM for BOSTON ----------------------------------------------------------


library(MASS)
data(Boston)

set.seed(12969599)
index <- sample(nrow(Boston),nrow(Boston)*0.75)
sample_index <- index
boston.train <- Boston[index,]
boston.test <- Boston[-index,]


#install.packages("mgcv")
library(mgcv)
Boston.gam <- gam(medv ~
                    s(crim)+s(zn)+s(indus)+chas+s(nox)+s(rm)+s(age)
                  +s(dis)+rad+s(tax)+s(ptratio)+s(black)+s(lstat),
                  data = boston.train)
summary(Boston.gam)

plot(Boston.gam, shade = TRUE, seWithMean = TRUE, scale = 0, pages = 1)


Boston.gam.predict.test <- predict(Boston.gam,
                                   boston.test) 
mean((predict(Boston.gam) - boston.train$medv)^2)
# 7.190695
mean((Boston.gam.predict.test -
        boston.test[, "medv"])^2) 
# 16.738

