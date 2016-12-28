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


#Convert all chars to factors
combi <- combi %>%
  mutate_if(is.character,as.factor)

#There are also some variables which are marked as int but are actually categorical. So we will
#Convert those also to factors
combi$MSSubClass <- as.factor(combi$MSSubClass)
combi$HouseStyle <- as.factor(combi$HouseStyle)
combi$OverallQual <- as.factor(combi$OverallQual)
combi$OverallCond <- as.factor(combi$OverallCond)


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

combi_kNN <- combi[,!colnames(combi) %in% c("SalePrice","IsTrain")] #Removed these three columns

combi_knn_imputed <- kNN(data=combi_kNN,k=54) #Not giving variable names,so it will impute NAs in all vars. k=sqrt(2919)

sapply(combi_knn_imputed,function(x){sum(ifelse(is.na(x),1,0))})
#All NAs have been removed

#Additional variables have been created. We can remove those

combi_knn_imputed <- subset(combi_knn_imputed,select = MSSubClass:AgeGarageBlt)
str(combi_knn_imputed)

combi <- data.frame(combi_knn_imputed,IsTrain=combi$IsTrain) 
sapply(combi,function(x){sum(ifelse(is.na(x),1,0))})

numericVariables <-colnames(combi[lapply(combi,class)=='integer' | lapply(combi,class)=='numeric'])
fctVariables <- colnames(combi[lapply(combi,class)=='factor']) #List of Factor variables

nzv <- nearZeroVar(combi,saveMetrics = TRUE)

correlations <- cor(subset(combi,IsTrain==TRUE, select=numericVariables))
corrplot(correlations, order = "hclust",type = "lower")

highCorr <- findCorrelation(correlations, cutoff = .75) #Variables which can be removed because of corelation. See Applied Predictive MOdeling. Pg. 47,56
colnames(combi[,highCorr])

combi <- combi[,-highCorr]

#Doing the preprocessing on combi. Should it be done on train + test separately???
#"BoxCox", "center", "scale" on numeric variables
preproc_numeric <- preProcess(combi[,numericVariables],method = c("BoxCox", "center", "scale"))
combi_preProc_numeric <- predict(preproc_numeric,combi[,numericVariables])

combi_dmy_factor <- dummy.data.frame(data = combi,names=fctVariables, all=FALSE) #We only need the dummyfied factor variables. Numeric variables will be added

combi_preProc <- data.frame(IsTrain=combi$IsTrain, SalePrice =rbind(train$SalePrice,test$SalePrice),combi_preProc_numeric,combi_dmy_factor)

str(combi_preProc)


#Linear Model
#First divide test into two and validate
train_1 <- subset(combi_preProc, IsTrain==TRUE)
nrow(train_1)

train_tr <- train_1[1:1100,]
train_tst <- train_1[1101:1460,]

train_tr$IsTrain=NULL
train_tst$IsTrain=NULL

train_tst_SalePrice <- train_tst$SalePrice
train_tst$SalePrice <- NULL

linear_model <- lm(SalePrice~. , data=train_tr)
train_tst_SalePrice_predict <- predict(linear_model,train_tst)

summary(linear_model)
