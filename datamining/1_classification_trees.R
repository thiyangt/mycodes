## packages
library(rpart)
library(tidyverse)
library(rattle)	#fancyRpartPlot
library(rpart.plot) #rpart.plot

## data
hn <- 1:24
income <- c(60, 85.5,64.8, 61.5,87,110.1,
            108,82.8,69,93, 51, 81, 75, 52.8,
            64.8, 43.2, 84,49.2, 59.4,
            66,47.4,33, 51,
            63)
length(income)

lotsize <- c(18.4, 16.8, 21.6, 20.8, 23.6,
             19.2, 17.6, 22.4, 20.0, 20.8, 22, 20,
             19.6, 20.8, 17.2, 20.4, 17.6, 17.6,
             16, 18.4,16.4, 18.8, 14, 14.8)
length(lotsize)
ownership <- c(rep("owner", 12),
               rep("nonowner", 12))

df <- data.frame(hn, income, lotsize, ownership)
df

## classification tree
df_rp <- rpart(ownership~lotsize+income, data=df)
df_rp

sort(income)
(59.4 + 60)/2

ggplot(data=df, aes(x=income, y=lotsize, col=ownership)) +
  geom_point() +
  geom_vline(aes(xintercept=59.7))

fancyRpartPlot(df_rp)
rpart.plot(df_rp)

## gini index
df_rp
## Gini left
# classes(k) - owner, nonowner
1-((7/8)^2+(1/8)^2)
## Gini right
1-((11/16)^2 + (5/16)^2)

## Entropy
## entropy left
-(7/8)*log((7/8), base=2) - (1/8)*log((1/8), base=2)
## entropy right
-(11/16)*log((11/16), base=2) - (5/16)*log((5/16), base=2)

## Combined impurity measures
## Gini
(8/24)*(0.219) + (16/24)*(0.430) #0.359
# Comment: before the split gini indexis 0.5
# After split it decreases


## Entropy
(8/24)*(0.544) + (16/24)*(0.896)
## Comment: before split entropy 1

## After split reduction of impurity measures

## To see the next split
df_rp$splits


## goodness of fit
printcp(df_rp, digits=3)

## tree 2
df_rp2 <- rpart(ownership~lotsize+income, data=df, 
               control = rpart.control(minbucket = 1))
df_rp2
fancyRpartPlot(df_rp2)

#########################
# Lesson 2: carot package

library(caret)

Titanic <- read.csv("https://raw.githubusercontent.com/mariocastro73/ML2020-2021/master/datasets/Titanic.csv")
summary(Titanic)


Titanic <- Titanic[,c("PClass","Age","Sex","Survived")]
Titanic$Survived <- as.factor(ifelse(Titanic$Survived==0,"Died","Survived"))
Titanic$PClass <- as.factor(Titanic$PClass)
Titanic$Sex <- as.factor(Titanic$Sex)
str(Titanic)
summary(Titanic)

Titanic <- na.omit(Titanic)
set.seed(9999)
# Cross Validation
## 80% data for training
## However, createDataPartition() tries to ensure a split that has 
## a similar distribution of the supplied variable in both datasets
train <- createDataPartition(Titanic[,"Survived"],p=0.8,list=FALSE) 
Titanic.trn <- Titanic[train,]
Titanic.tst <- Titanic[-train,]

ctrl  <- trainControl(method  = "cv",number  = 10) #, summaryFunction = multiClassSummary

fit.cv <- train(Survived ~ ., data = Titanic.trn, method = "rpart",
                trControl = ctrl, 
                # preProcess = c("center","scale"), 
                # tuneGrid =data.frame(cp=0.05))
                tuneLength = 30) # metric="Kappa",

## obtain cross validation results for all sample
fit.cv$resample
mean(fit.cv$resample$Accuracy)
getTrainPerf(fit.cv)



## Out of sample prediction
pred <- predict(fit.cv,Titanic.tst)
confusionMatrix(table(Titanic.tst[,"Survived"],pred))
print(fit.cv)
plot(fit.cv)
plot(fit.cv$finalModel)
text(fit.cv$finalModel)


library(rpart.plot)
rpart.plot(fit.cv$finalModel)
rpart.plot(fit.cv$finalModel,fallen.leaves = FALSE)

############
# Boston house price data
# Load packages
library(mlbench)
library(caret)
library(corrplot)

# Load data
data(BostonHousing)

set.seed(101)
sample <- createDataPartition(BostonHousing$medv, p=0.80, list = FALSE)
train <- BostonHousing[sample,]
test <- BostonHousing[-sample,]
train$chas <- as.numeric(train$chas)
str(train$chas)

control <- trainControl(method='cv', number=10)
metric <- 'RMSE'

set.seed(101)
fit.lm <- train(medv ~., data=train, method='lm', metric=metric, trControl=control)
set.seed(101)
fit.glm <- train(medv~., data=train, method='glm', metric=metric, 
                  trControl=control)
fit.cart <- train(medv~., data=train, method='rpart', metric=metric, 
                   trControl=control)
rpart.plot(fit.cart, type = 3, box.palette = c("red", "grey"), fallen.leaves = TRUE)

# Compare the results of these algorithms
boston.results <- resamples(list(lm=fit.lm, 
                                 glm=fit.glm, 
                                 cart=fit.cart))

# Summary and Plot of Results
summary(boston.results)
dotplot(boston.results)
