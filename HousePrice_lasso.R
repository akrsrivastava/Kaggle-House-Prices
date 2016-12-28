rm(list = ls(all = TRUE))
library(dplyr)
library(ggplot2)
library(gridExtra)
library(VIM)
library(stringr)
library(caret)
library(dummies)
library(Boruta)
library(xgboost)
library(corrplot)
library(VIF)
library(glmnet)
train <- read.csv("D:/amit/Data Science/Kaggle/House Prices/train.csv", stringsAsFactors=FALSE)
test <- read.csv("D:/amit/Data Science/Kaggle/House Prices/test.csv", stringsAsFactors=FALSE)
str(train)
#All variables are either Int or Char
#Target Variable is Sale Price


test$SalePrice=NA
train$IsTrain=TRUE
test$IsTrain=FALSE
combi <- rbind(train,test)
test_Id=test$Id


str(train$Id)


#Analysis of blanks and NAs
#Check NAs
sapply(combi,function(x){sum(ifelse(is.na(x),1,0))})
#There are a lot of Variables where NA actually means something. See the data description. Lets analyze them first

#Alley
nrow(combi[is.na(combi$Alley),])
#2721 NAs As per data dictionary NA means no alley/ So lets update NAs to No.
combi$Alley <- ifelse(is.na(combi$Alley),"NO",combi$Alley)

#BsmtQual
nrow(combi[is.na(combi$BsmtQual),])
#81 NA's
combi$BsmtQual <- ifelse(is.na(combi$BsmtQual),"NO",combi$BsmtQual)

#BsmtCond
combi$BsmtCond <- ifelse(is.na(combi$BsmtCond),"NO",combi$BsmtCond)

#BsmtExposure
combi$BsmtExposure <- ifelse(is.na(combi$BsmtExposure),"NO",combi$BsmtExposure)

#BsmtFinType1
combi$BsmtFinType1 <- ifelse(is.na(combi$BsmtFinType1),"NO",combi$BsmtFinType1)

#BsmtFinType2
combi$BsmtFinType2 <- ifelse(is.na(combi$BsmtFinType2),"NO",combi$BsmtFinType2)

#FireplaceQu
combi$FireplaceQu <- ifelse(is.na(combi$FireplaceQu),"NO",combi$FireplaceQu)

#GarageType
combi$GarageType <- ifelse(is.na(combi$GarageType),"NO",combi$GarageType)

#GarageFinish
combi$GarageFinish <- ifelse(is.na(combi$GarageFinish),"NO",combi$GarageFinish)

#GarageQual
combi$GarageQual <- ifelse(is.na(combi$GarageQual),"NO",combi$GarageQual)

#GarageCond
combi$GarageCond <- ifelse(is.na(combi$GarageCond),"NO",combi$GarageCond)

#PoolQC
combi$PoolQC <- ifelse(is.na(combi$PoolQC),"NO",combi$PoolQC)

#Fence
combi$Fence <- ifelse(is.na(combi$Fence),"NO",combi$Fence)

#MiscFeature
combi$MiscFeature <- ifelse(is.na(combi$MiscFeature),"NO",combi$MiscFeature)

#Done. Now check how many NAs left
sapply(combi,function(x){sum(ifelse(is.na(x),1,0))})

combi$Utilities= as.numeric(str_replace_all(combi$Utilities,c("ELO"=1,"NoSeWa"=2,"NoSewr"=3,"AllPub"=4)))
combi$ExterQual= as.numeric(str_replace_all(combi$ExterQual,c("Po"=1,"Fa"=2,"TA"=3,"Gd"=4,"Ex"=5)))
combi$ExterCond= as.numeric(str_replace_all(combi$ExterCond,c("Po"=1,"Fa"=2,"TA"=3,"Gd"=4,"Ex"=5)))
combi$BsmtQual= as.numeric(str_replace_all(combi$BsmtQual,c("NO"=0,"Po"=1,"Fa"=2,"TA"=3,"Gd"=4,"Ex"=5)))
combi$BsmtCond= as.numeric(str_replace_all(combi$BsmtCond,c("NO"=0,"Po"=1,"Fa"=2,"TA"=3,"Gd"=4,"Ex"=5)))
combi$BsmtExposure= as.numeric(str_replace_all(combi$BsmtExposure,c("NO"=0, "No"=0, "Mn"=1,"Av"=2,"Gd"=3)))
combi$BsmtFinType1= as.numeric(str_replace_all(combi$BsmtFinType1,c("NO"=0, "Unf"=1,"LwQ"=2,"Rec"=3,"BLQ"=4,"ALQ"=5,"GLQ"=6)))
combi$BsmtFinType2= as.numeric(str_replace_all(combi$BsmtFinType2,c("NO"=0, "Unf"=1,"LwQ"=2,"Rec"=3,"BLQ"=4,"ALQ"=5,"GLQ"=6)))
combi$HeatingQC= as.numeric(str_replace_all(combi$HeatingQC,c("Po"=1,"Fa"=2,"TA"=3,"Gd"=4,"Ex"=5)))
combi$KitchenQual= as.numeric(str_replace_all(combi$KitchenQual,c("Po"=1,"Fa"=2,"TA"=3,"Gd"=4,"Ex"=5)))
combi$FireplaceQu= as.numeric(str_replace_all(combi$FireplaceQu,c("Po"=1,"Fa"=2,"TA"=3,"Gd"=4,"Ex"=5,"NO"=0)))
combi$GarageFinish= as.numeric(str_replace_all(combi$GarageFinish,c("NO"=0, "Unf"=1,"RFn"=2,"Fin"=3)))
combi$GarageQual= as.numeric(str_replace_all(combi$GarageQual,c("NO"=0,"Po"=1,"Fa"=2,"TA"=3,"Gd"=4,"Ex"=5)))
combi$GarageCond= as.numeric(str_replace_all(combi$GarageCond,c("NO"=0,"Po"=1,"Fa"=2,"TA"=3,"Gd"=4,"Ex"=5)))
combi$PoolQC= as.numeric(str_replace_all(combi$PoolQC,c("NO"=0,"Fa"=1,"TA"=2,"Gd"=3,"Ex"=4)))
combi$Fence= as.numeric(str_replace_all(combi$Fence,c("NO"=0,"MnWw"=1,"GdWo"=2,"MnPrv"=3,"GdPrv"=4)))



#Convert all chars to factors
combi <- combi %>%
  mutate_if(is.character,as.factor)

#There are also some variables which are marked as int but are actually categorical. So we will
#Convert those also to factors
combi$MSSubClass <- as.factor(combi$MSSubClass)


#Age of property
#Age of property will be a better indicator than year built and year sold #Assuming month of year built is 1
combi$AgeBuilt <- (combi$YrSold-combi$YearBuilt)*12 +combi$MoSold


#Change other years into age
combi$AgeRemodAdd <- (combi$YrSold-combi$YearRemodAdd)*12
combi$AgeGarageBlt <- (combi$YrSold-combi$GarageYrBlt)*12

#Drop the YrSold,YearBuilt,"YearRemodAdd","GarageYrBlt" and MoSold variables
combi[,c("YrSold","YearBuilt","MoSold","YearRemodAdd","GarageYrBlt")] <- NULL


#Drop ID
combi$Id <- NULL





# Right now, I will do a quick kNN imputation to set a benchmark

combi_kNN <- combi[,!colnames(combi) %in% c("SalePrice")] #Will not impute these cols.

combi_knn_imputed <- kNN(data=combi_kNN,k=54) #Not giving variable names,so it will impute NAs in all vars. k=sqrt(2919)

sapply(combi_knn_imputed,function(x){sum(ifelse(is.na(x),1,0))})
#All NAs have been removed

#Additional variables have been created. We can remove those

combi_knn_imputed <- subset(combi_knn_imputed,select = MSSubClass:AgeGarageBlt)
str(combi_knn_imputed)

numericVariables <-colnames(combi_knn_imputed[lapply(combi_knn_imputed,class)=='integer' | lapply(combi_knn_imputed,class)=='numeric'])


correlations <- cor(subset(combi_knn_imputed,IsTrain==TRUE, select=numericVariables))
corrplot(correlations, order = "hclust",type = "lower")




highCorr <- findCorrelation(correlations, cutoff = .75) #Variables which can be removed because of corelation. See Applied Predictive MOdeling. Pg. 47,56
colnames(combi_knn_imputed[,highCorr])

combi_knn_imputed <- combi_knn_imputed[,-highCorr]

preprocess <- preProcess(combi_knn_imputed,method="scale")
combi_knn_imputed <- predict(preprocess, combi_knn_imputed)

combi_knn_imputed_dmy <- dummy.data.frame(data=combi_knn_imputed,dummy.classes = "factor",all=TRUE)


# #Lasso on a subset of train data
train_set <- combi_knn_imputed_dmy[which(combi_knn_imputed_dmy$IsTrain==TRUE),]
train_set <- combi_knn_imputed_dmy[1:1100,]
#train_set <- train_set[]

test_set <- combi_knn_imputed_dmy[1101:1460,]
#test_set <- test_set["IsTrain"==FALSE,]


# Lasso alpha =1 for Lasso
#Cross validation to find optimal Lambda
CV=cv.glmnet(x=as.matrix(train_set),y=log(combi[1:1100,"SalePrice"]), family="gaussian",alpha=1,nlambda=100)
plot(CV)
CV


lasso_model <-  glmnet(x=as.matrix(train_set) ,y=log(combi[1:1100,"SalePrice"]), family="gaussian", alpha=1, lambda=CV$lambda.1se)

lasso_model$lambda
print(lasso_model)


lasso_model_predict <- predict.glmnet(lasso_model, as.matrix(test_set) )

checkResult <- data.frame(Actual=combi[1101:1460,"SalePrice"], Predicted=exp(lasso_model_predict))
colnames(checkResult) <- c("Actual","SalePrice")
RMSE(checkResult$Actual,checkResult$SalePrice)

# RIdge alpha=0 for ridge
#Cross validation to find optimal Lambda
CV=cv.glmnet(x=as.matrix(train_set),y=log(combi[1:1100,"SalePrice"]), family="gaussian",alpha=0,nlambda=100)
plot(CV)
CV


lasso_model <-  glmnet(x=as.matrix(train_set) ,y=log(combi[1:1100,"SalePrice"]), family="gaussian", alpha=0, lambda=CV$lambda.1se)

lasso_model$lambda
print(lasso_model)


lasso_model_predict <- predict.glmnet(lasso_model, as.matrix(test_set) )

checkResult <- data.frame(Actual=combi[1101:1460,"SalePrice"], Predicted=lasso_model_predict)
colnames(checkResult) <- c("Actual","SalePrice")
RMSE(checkResult$Actual,checkResult$SalePrice)

#CAret for Elastic Net
cv.ctrl <- trainControl(method = "repeatedcv", repeats = 1,number = 5, 
                        allowParallel=T)

cv.grid <- expand.grid(alpha = seq(0.01,1,0.01),
                       lambda=seq(0.01,1,0.01)
)


Sys.time()
cv.train = train(
  x=as.matrix(train_set),
  y=log(combi[1:1100,"SalePrice"]),
  trControl = cv.ctrl,
  tuneGrid = cv.grid ,
  method = "glmnet"
)
Sys.time()
cv.train #

lasso_model_predict <- predict(cv.train, as.matrix(test_set) )
checkResult <- data.frame(Actual=combi[1101:1460,"SalePrice"], Predicted=lasso_model_predict)
colnames(checkResult) <- c("Actual","SalePrice")
RMSE(checkResult$Actual,checkResult$SalePrice)

##USE THE ENTIRE TRAIN SET
#Lasso on complete train data
train_set <- combi_knn_imputed_dmy[which(combi_knn_imputed_dmy$IsTrain==TRUE),]

test_set <- combi_knn_imputed_dmy[which(combi_knn_imputed_dmy$IsTrain==FALSE),]




lm <- glmnet(x=as.matrix(train_set) ,y=combi[1:1460,"SalePrice"], family="gaussian", alpha=1)
plot(lm,xvar="lambda")

#Cross validation to find optimal Lambda
CV=cv.glmnet(x=as.matrix(train_set),y=log(combi[1:1460,"SalePrice"]), family="gaussian",alpha=1,nlambda=100) #alpha=1 for lasso, 0 for ridge
plot(CV)
CV

coef_df <- as.data.frame(as.matrix(coef(CV)))
coef_df <- data.frame("Var" = rownames(coef_df),"Coef"= coef_df[,"1"])

coef_df <- coef_df[order(coef_df$Coef,decreasing = TRUE),]
head(coef_df,40)
ggplot(subset(coef_df,abs(coef_df$Coef) >0)) + geom_bar(aes(x=Var,y=Coef),stat="identity", fill="orange") +
  coord_flip()
lasso_model <-  glmnet(x=as.matrix(train_set) ,y=log(combi[1:1460,"SalePrice"]), family="gaussian", alpha=1, lambda=CV$lambda.1se)

lasso_model$lambda
plot(lm,xvar="dev")


lasso_model_predict <- predict.glmnet(lasso_model, as.matrix(test_set) )


#Ridge regression on the entire train set
#Lasso on complete train data
train_set <- combi_knn_imputed_dmy[which(combi_knn_imputed_dmy$IsTrain==TRUE),]

test_set <- combi_knn_imputed_dmy[which(combi_knn_imputed_dmy$IsTrain==FALSE),]

#Cross validation to find optimal Lambda
CV=cv.glmnet(x=as.matrix(train_set),y=log(combi[1:1460,"SalePrice"]), family="gaussian",alpha=0,nlambda=100) #alpha=1 for lasso, 0 for ridge
plot(CV)
CV

lasso_model <-  glmnet(x=as.matrix(train_set) ,y=log(combi[1:1460,"SalePrice"]), family="gaussian", alpha=0, lambda=CV$lambda.1se)

lasso_model$lambda
plot(lasso_model,xvar="dev")


lasso_model_predict <- predict.glmnet(lasso_model, as.matrix(test_set) )

#Caret cross validation for Elastic Net
# pack the training control parameters
cv.ctrl <- trainControl(method = "repeatedcv", repeats = 1,number = 5, 
                        allowParallel=T)

cv.grid <- expand.grid(alpha = seq(0.01,1,length=15),
                       lambda=seq(0.01,1,length=15)
)


Sys.time()
cv.train = train(
  x=as.matrix(train_set),
  y=log(combi[1:1460,"SalePrice"]),
  trControl = cv.ctrl,
  tuneGrid = cv.grid ,
  method = "glmnet"
)
Sys.time()
cv.train #The final values used for the model were alpha = 0.01 and lambda = 0.1514286. 

lasso_model_predict <- predict(cv.train, as.matrix(test_set) )

# checkResult <- data.frame(Actual=combi[1101:1460,"SalePrice"], Predicted=lasso_model_predict)
# RMSE(checkResult$Actual,checkResult$s0)

response_df <- data.frame(Id=test$Id,SalePrice=exp(lasso_model_predict))
colnames(response_df) <- c("Id","SalePrice")
str(response_df)
write.csv(response_df,file="D:/amit/Data Science/Kaggle/House Prices/Log_Ridge_0410.csv",row.names=FALSE)
