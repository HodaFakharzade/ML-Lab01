#-------------------------------------------------------
# A.1.1
#-------------------------------------------------------
library(tidyverse)

Data <- read.csv("Data/optdigits.csv",header = FALSE)
#intended to represent values of a categorical variable
Data$V65 <- as.factor(Data$V65) 
n <- dim(Data)[1] # Number of rows(Samples) = 3822
set.seed(1234)
id=sample(1:n, floor(n*0.5))
train=Data[id,]
id1=setdiff(1:n, id)
set.seed(12345)
id2 <- sample(id1,floor(n*0.25))
valid <- Data[id2,] #Validation Set
id3 <- setdiff(id1,id2) #Test Set
test <- Data[id3,]

#-------------------------------------------------------
# A.1.2
#-------------------------------------------------------

library(kknn)

kknn_classifier_train <- kknn(V65~.,train = train, test = train,
                              kernel = "rectangular", k = 30)
summary(kknn_classifier_train)

kknn_classifier_test <- kknn(V65~.,train = train, test = test,
                              kernel = "rectangular", k = 30)
summary(kknn_classifier_test)

pred_train <- kknn_classifier_train$fitted.values
pred_test <- kknn_classifier_test$fitted.values
temp <- table(train$V65, pred_train)

temp2 <- table(test$V65,pred_test )
print("confusion matrix for Train ")
temp
print("confusion matrix for test ")
temp2

#missclassificationError From Lecture 1 slides

missclass=function(X,X1){
  n=length(X)
  return(1-sum(diag(table(X,X1)))/n)
}
print("missclassification Error for Train Data")
missclass(train$V65,pred_train) 

print("missclassification Error for Test Data")

missclass(test$V65,pred_test)

# overall performance of the model
acc=function(x,x2)
{
  n=length(x)
  ac=sum(diag(table(x,x2)))/n
  return(ac)
}
print("accuracy on training Data ")
acc(train$V65,pred_train)

print("accuracy on test set ")
acc(test$V65,pred_test)
#-------------------------------------------------------
# A.1.3
#-------------------------------------------------------
#any 2 cases of digit “8” in the training data which were
#easiest to classify and 3 cases that were hardest to classify

v=data.frame(kknn_classifier_train$prob)
prob <- colnames(v)[apply(v, 1, which.max)]
v$y<-train$V65
v$fit <- pred_train
v$prob <- prob
pred_8 <- v[v$y == 8,]

###
# Best
best <- as.numeric(row.names(pred_8[order(-pred_8[,9]),][1:2,]))
best

# Worse cases 
Worse <- as.numeric(row.names(pred_8[order(pred_8[,9]),][1:3,]))
Worse
#best cases visalization
col=heat.colors(12)

heatmap(t(matrix(unlist(train[best[1],-65]), nrow=8)),Colv = NA, Rowv = NA,col=c("black","white" ))

heatmap(t(matrix(unlist(train[best[2],-65]), nrow=8)), Colv = NA, Rowv = NA,col=c("black","white" ))

#worst cases visalization
#Reshape features for each of these cases as matrix 8x8 and visualize the corresponding digits

heatmap(t(matrix(unlist(train[Worse[1],-65]), nrow=8)), Colv = NA, Rowv = NA,col=c("black","white"))

heatmap(t(matrix(unlist(train[Worse[2],-65]), nrow=8)), Colv = NA, Rowv = NA,col=c("black","white"))

heatmap(t(matrix(unlist(train[Worse[3],-65]), nrow=8)), Colv = NA, Rowv = NA,col=c("black","white"))
#-------------------------------------------------------
# A.1.4
#-------------------------------------------------------
library(reshape)
library(data.table)
library(ggplot2)

missclass_train <- vector()
missclass_val <- vector()



for (i in 1:30) {
  
  # fit model
  train_model <- kknn(formula =V65~., kernel = "rectangular", train = train, 
                      test = train, k = i)
  val_model <- kknn(formula = V65~., kernel = "rectangular", train = train, 
                    test = valid, k = i)
  
  # confusion matrix
  train_c_table <- table(train$V65, train_model$fitted.values)
  val_c_table <- table(valid$V65  , val_model$fitted.values)
  
  # misclassification rates
  missclass_train <- c(missclass_train, 1-sum(diag(train_c_table))/sum(train_c_table))
  missclass_val <- c(missclass_val, 1-sum(diag(val_c_table))/sum(val_c_table))

}

missclass <- melt(data.table(k = 1:30, Training = missclass_train, Validation = missclass_val), "k",
                  variable.name = "Legend",)


# plot misclassification rates and cross-entropy by 'k' for training and validation datasets
ggplot(missclass) + geom_line(aes(k, value, colour = Legend)) + theme_bw() + 
  geom_vline(xintercept = 3, linetype = "dotted") + scale_x_continuous(breaks = 1:30) +
  xlab(" k- from 1:30 ") + ylab("Misclassification Error") + 
  ggtitle("Finding Best K in KNN")
#-------------------------------------------------------
# A.1.5
#-------------------------------------------------------

#compute the empirical risk for the validation data as cross-entropy ( when computing log of probabilities add a small constant within log, e.g. 1e-15

cross_entropy_train <- c()
cross_entropy_val <- c()
num_classes <- length(unique(train$V65))
num_train_examples <- nrow(train)
num_val_examples <- nrow(valid)
one_hot_y_train <- t(sapply(as.numeric(train$V65), 
                            function(x) {c(rep(0, x - 1), 1, rep(0, num_classes - x))}))
one_hot_y_val <- t(sapply(as.numeric(valid$V65), 
                          function(x) {c(rep(0, x - 1), 1, rep(0, num_classes - x))}))

for (i in 1:30) {
  
  # fit model
  train_model <- kknn(V65~., kernel = "rectangular", train = train, test = train, k = i)
  val_model <- kknn(V65~., kernel = "rectangular", train = train, test = valid, k = i)
  
  
  
  
  # cross-entropy
  cross_entropy_train <- c(cross_entropy_train, sum(one_hot_y_train * -log(train_model$prob + 10^-15))/num_train_examples)
  cross_entropy_val <- c(cross_entropy_val, sum(one_hot_y_val * -log(val_model$prob + 10^-15))/num_val_examples)
  
}
cross_entropy <- melt(data.table(k = 1:30,  Validation = cross_entropy_val),
                      "k", variable.name = " ")

ggplot(cross_entropy) + geom_line(aes(k, value),col="red") + theme_bw() + 
  geom_vline(xintercept = 6, linetype = "dotted") + scale_x_continuous(breaks = 1:30) +
  xlab("Hyper-Parameter 'k'") + ylab("Cross-Entropy") + 
  ggtitle("Finding the Optimal Hyper-Parameter")



