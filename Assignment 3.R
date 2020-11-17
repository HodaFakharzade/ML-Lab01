### 932A99 - Lab 1 ###

library(glmnet)
library(kknn)


#### Assignment 3 ####

tecator <- read.csv("tecator.csv")
names(tecator)[1] <- "Sample"
n_tecator <- nrow(tecator)
set.seed(12345)
id <- sample(1:n_tecator, floor(n_tecator*0.5))
train_data <- tecator[id,]
test_data <- tecator[-id,]

## 3.1 - Linear Regression

train <- train_data[-c(1,103,104)]
test <- test_data[-c(1,103,104)]

# p(y|x,w) = N(w^Tx, sigma^2)

# where w = {w0,...,w100}
#       x = {1,Channel1,...,Channel100}


fat_lm <- lm(formula = Fat~.,
             data = train)

summary(fat_lm)

fat_lm_pred <- predict(fat_lm,
                       newdata = test)

test_resids <- test$Fat - fat_lm_pred
test_SSE <- sum(test_resids^2)
sqrt(test_SSE/(length(test_resids)-length(fat_lm$coefficients)))

# The training error is very low at 0.3191 but for the test data
# it is 105.57, possibly suggesting that the model is overfitted for 
# the training data. 

## 3.2 - Objective function to optimize in LASSO

# Sum (from i to N) (yi - w0 - w1x1i - ... - wpxpi)^2
# subject to sum (from j to p) |wj| leq s

## 3.3 - Fit LASSO model

fat_lasso <- glmnet(as.matrix(train[,1:100]), 
                    train[,101],
                    alpha = 1)

plot(fat_lasso,
     xvar = "lambda")
print(fat_lasso)

# As the value of log(lambda) goes up the amount of features in the model goes down as 
# the coefficients are forced to 0. At the very first step, when lambda goes from 0 to
# 0.0051 there is a very large reduction in features, going from the original 100 down 
# to only 37. 

# The last remaining feature is Channel41.
# The model goes from four to three features at lambda=0.7082 and remains at three until 0.9362

## 3.4 - Degrees of freedom vs Lambda

t <- print(fat_lasso)
plot(t$Lambda, 
     t$Df,
     xlab = "Lambda",
     ylab = "Df")

# The plot shows that as lambda increases the degrees of freedom go down, which is 
# expected since the degrees of freedom is the amount of features in the model and 
# as lambda increases more of these features are forced to 0. 

## 3.5 - Ridge regression

fat_ridge <- glmnet(as.matrix(train[,1:100]), 
                    train[,101],
                    alpha = 0)

plot(fat_ridge, 
     xvar = "lambda")

# It can be observed that the coefficients are forced towards 0 but at a much
# slower and more gradual rate compared to for the LASSO model. It is also the 
# case that for the Ridge model no feature is completely removed from the model,
# but are instead kept with the very low coefficient. This can be observed by
# the axis ticks at the top of the graph, which always say 100, compared to
# for the LASSO model where this number goes down as features are removed. 

## 3.6 - Optimal LASSO model with cross-validation

# 10 folds as chosen by default
fat_lasso_cv <- cv.glmnet(as.matrix(train[,1:100]), 
                          train[,101],
                          alpha = 1)

plot(fat_lasso_cv)
# It can be observed that as log(lambda) increases so does
# the Mean-Squared error, at seemingly an exponential rate up until
# log(lambda)=0 where the increase halts. After this point the MSE
# once again starts growing, slowly at first, at what could be seen
# as being an exponential rate. 

print(fat_lasso_cv)
# Optimal lambda is 0.05745 and the amount of chosen variables is 8.

# The print function also returns that the largest lambda where MSE
# is still within 1 standard error of the minimum error is 0.09147.
# Compared to this lambda value, log(lambda)=-2 -> lambda = 0.1353, and
# as such would suggest that the error for log(lambda)=-2 is more than 
# 1 standard deviation away from the minimum value. 

fat_lasso_cv_preds <- predict(fat_lasso_cv,
                              newx = as.matrix(test[,1:100]),
                              s = "lambda.min")

plot(x = fat_lasso_cv_preds, 
     y = test$Fat,
     xlab = "Predicted test values",
     ylab = "Test values",
     main = "Test values vs predicted values")

# Overall it seems the model looks to be making fairly solid predictions, 
# at least for lower levels of Fat. The correlation between the predicted
# values and the real values is 0.96. It can be seen that as Fat levels 
# increase there is a slight curve to the slope of the points, as the model
# predicts too high levels of fat. As an example of this there are 5 points
# predicted to have a fat level between 50 and 60 while in reality these 
# observations all had levels below 50. 

## 3.7 - Generate new data and compare

train_fits <- predict(fat_lasso_cv,
                      newx = as.matrix(train[,1:100]),
                      s = "lambda.min")
train_resid <- train$Fat - train_fits
train_sigma <- sd(train_resid)

new_data <- 0
set.seed(12345)
for (obs in 1:108) {
  intercept <- coef(fat_lasso_cv, s ="lambda.min")[1]
  wx <- sum(coef(fat_lasso_cv, s ="lambda.min")[-1] * test[obs,1:100])
  new_data[obs] <- rnorm(n = 1,
                         mean = (intercept + wx),
                         sd = train_sigma)
}

plot(x = new_data, 
     y = test$Fat,
     xlab = "Generated test values",
     ylab = "Test values",
     main = "Test values vs generated data")

# In general conclusions that can be drawn from this plot are similar
# to those from the predicted values in that the generated data
# appears to follow the true data fairly well, in particular for 
# lower levels of fat. The correlation between the true data and the
# generated data is 0.93, which is slightly lower than for the 
# predicted values. This makes sense as it can be observed that for the
# generated data there are now points that go above 60, whereas for 
# the predicted value the upper boundary was around 60. 