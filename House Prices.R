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
train <- read.csv("D:/amit/Data Science/Kaggle/House Prices/train.csv", stringsAsFactors=FALSE)
test <- read.csv("D:/amit/Data Science/Kaggle/House Prices/test.csv", stringsAsFactors=FALSE)
str(train)
#All variables are either Int or Char
#Target Variable is Sale Price
#It Seems all Char variables should be changed to Factors
# train <- train %>%
#           mutate_if(is.character,as.factor)
# 
# test <- test %>%
#   mutate_if(is.character,as.factor)

test$SalePrice=NA
train$IsTrain=TRUE
test$IsTrain=FALSE
combi <- rbind(train,test)
test_Id=test$Id


str(train$Id)
#fctVariables <- colnames(combi[lapply(combi,class)=='factor']) #43 Factor variables
# intVariables <- colnames(combi[lapply(combi,class)=='integer' | lapply(combi,class)=='numeric']) #38 int variables


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

#MSZoning has 4
combi[is.na(combi$MSZoning),]

ggplot(combi,aes(factor(MSZoning),factor(Neighborhood))) + geom_point()
head(combi[combi$Neighborhood=="IDOTRR",c("MSZoning","MSSubClass")],20)

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
combi$AgeInMonths <- (combi$YrSold-combi$YearBuilt)*12 +combi$MoSold
#Drop the YrSold,YearBuilt and MoSold variables
combi[,c("YrSold","YearBuilt","MoSold")] <- NULL

#Drop ID
combi$Id <- NULL




# Right now, I will do a quick kNN imputation to set a benchmark

combi_kNN <- combi[,!colnames(combi) %in% c("SalePrice","IsTrain")] #Removed these three columns

combi_knn_imputed <- kNN(data=combi_kNN,k=54) #Not giving variable names,so it will impute NAs in all vars. k=sqrt(2919)

sapply(combi_knn_imputed,function(x){sum(ifelse(is.na(x),1,0))})
#All NAs have been removed
str(combi_knn_imputed)
#Additional variables have been created. We can remove those

combi_knn_imputed <- subset(combi_knn_imputed,select = MSSubClass:SaleCondition)
str(combi_knn_imputed)

combi <- data.frame(combi_knn_imputed,combi[,c("SalePrice","IsTrain")]) 

#Now check for any blanks
sapply(combi,function(x){sum(ifelse(str_trim(x)=="",1,0))}) #Check no. of blanks in the features
#No blanks. Everything fine.

#There are 23 integer variables. At first pass, the predicted linear regression were very high. I am thinking this may
#be due to outliers. Let me check for outliers in the numeric vars.
numVariables <- colnames(combi[lapply(combi,class)=='integer' | lapply(combi,class)=='numeric'])
fctVariables <- colnames(combi[lapply(combi,class)=='factor']) #46 Factor variables

# for (i in 1:length(nbrVariables))
# {
#   q10 <- quantile(combi[,nbrVariables[i]],0.1)
#   q90 <- quantile(combi[,nbrVariables[i]],0.9)
#   col <- c(col,nbrVariables[i])
#   
#   
#}
ggplot(combi) + geom_boxplot(aes(x="", y=LotFrontage) ) + labs(x="Lot Frontage" , y= "Value")

# combi1 <- combi[,numVariables]
# p <- list()
# for(i in 1:length(combi1)){
#   print(i)
#   p[[i]] <- ggplot(combi1) + geom_boxplot(aes(x="", y=combi1[,i]) ) +labs(x=names(combi1)[i],y='')
# }
# do.call(grid.arrange,p)
#For some reason it is showing up the same boxplot for all the features and I am not able to debug

#So I will delete all rows for each feature which do not lie in the 0.05 to 0.95 quantiles
# combi1 <- combi[,numVariables]
# combi1 <- combi1[,!colnames(combi1) %in% "SalePrice"]
# combi2 <- combi
# print(nrow(combi2))
# for (i in 1:length(combi1))
# {
#   
#   col_name <- colnames(combi1)[i]
#   print(col_name)
#   q1 <- quantile(combi1[,col_name],0.02)
#   q2 <- quantile(combi1[,col_name],0.98)
#   
#   combi2 <- subset(combi2, combi[[col_name]] > q1 & combi[[col_name]] < q2) #Look at how I need to use [[]] when using a variable name
#   # The fact that "[[" will evaluate its argument is why it is superior to "$" for programing
#   print(nrow(combi2))
# }
# boxplot(combi$LotFrontage)
# 
# rm(combi1)
# 
# combi3 <- left_join(x=combi2,y=combi[,!colnames(combi) %in% numVariables],by="Id") 

# combi1 <- combi[,numVariables]
# combi1 <- combi1[,!colnames(combi1) %in% "SalePrice"]
# for (i in 1:length(combi1))
# {
#   col_name <- colnames(combi1)[i]
#   print(col_name)
#   # print(combi[which(combi[[col_name]]==boxplot.stats(combi[,col_name],coef = 3)$out),col_name])
#   print(table(combi[[col_name]]))
# }

#hist(combi$MiscVal)


#First let us do Boruta Feature Analysis


boruta_features <- Boruta(data=subset(combi,combi$IsTrain==TRUE),SalePrice~.,doTrace=2)
arrange(cbind(attr=rownames(attStats(boruta_features)), attStats(boruta_features)),desc(medianImp))

impVars <- getSelectedAttributes(boruta_features)
#Conducting a baseline linear regression

#First we will test on a subsample of the Train dataset


train_lm <- combi[combi$IsTrain==TRUE,][1:1100,] #FIrst 1100 rowsvof train
test_lm <- combi[combi$IsTrain==TRUE,][1101:1460,] #Remaining rows of train


fctVariables_train_lm <- colnames(train_lm[lapply(train_lm,class)=='factor']) #List of Factor variables



train_lm$Id <- NULL

test_lm_Id <- test$Id
test_lm$Id <- NULL

test_lm_Sales <- test_lm$SalePrice
#test_lm$SalePrice <- NULL

train_lm$IsTrain <- NULL
test_lm$IsTrain <- NULL
#test_lm <- test_lm[,impVars]

modelVars <- paste(impVars,collapse = "+")
fml <- paste("log(SalePrice)~" , modelVars)
baseline_linear_model <- lm(data=train_lm,fml)



#baseline_linear_model_predict <- predict(baseline_linear_model,newdata = test_lm)
#Getting error Error in model.frame.default(Terms, newdata, na.action = na.action, xlev = object$xlevels) : 
#factor Condition2 has new levels RRAe
#This means that in the test set Condition2 has some levels which are not in the train set.
#We will need to change these level values to NA, if we want to use the linear model trained on the train
#http://stackoverflow.com/questions/4285214/predict-lm-with-an-unknown-factor-level-in-test-data
#Another option is to add this new level to the model via baseline_linear_model$xlevel 
#http://stackoverflow.com/questions/22315394/factor-has-new-levels-error-for-variable-im-not-using
# baseline_linear_model$xlevels[["Condition2"]] <- union(baseline_linear_model$xlevels[["Condition2"]],levels(test_lm$Condition2))
#baseline_linear_model_predict <- predict(baseline_linear_model,newdata = test_lm)
#Now its giving error for RoofStyle. Need to write a generic code

for (x in 1:length(baseline_linear_model$xlevels))
{
  #print (baseline_linear_model$xlevels[[x]])
  x1=baseline_linear_model$xlevels
  #print (names(x1)[x])
  #baseline_linear_model$xlevels[[(names(x1)[x])]]
  #print(levels(test_lm[,names(x1)[x]]))
  baseline_linear_model$xlevels[[(names(x1)[x])]] <- union(baseline_linear_model$xlevels[[(names(x1)[x])]],levels(test_lm[,names(x1)[x]]))
}
#Now Predict
summary(baseline_linear_model)

baseline_linear_model_predict <- predict(baseline_linear_model,test_lm)
train_lm_sales <- train_lm$SalePrice[1:1100]
train_lm$SalePrice <- NULL

train_xgBoost <- xgb.DMatrix(data.matrix (train_lm),label = train_lm_sales)
xgmodel <- xgboost(data=train_xgBoost,nrounds=1000)
xgmodel_predict <- predict(xgmodel,data.matrix(test_lm))
str(xgmodel_predict)
RMSE(xgmodel_predict,test_lm_Sales)
# str(baseline_linear_model_predict)
# RMSE(baseline_linear_model_predict,test_lm_Sales)
#RMSE(log(baseline_linear_model_predict),log(test_lm_Sales))
#plot(baseline_linear_model_predict,test_lm_Sales)
plot(test_lm_Sales,train_lm_sales)


#Now traing XGBoost on entire train
train_xgboost <- combi[combi$IsTrain==TRUE,]
test_xgboost <- combi[combi$IsTrain==FALSE,]
#fctVariables_train_lm <- colnames(train_lm[lapply(train_lm,class)=='factor']) #List of Factor variables


test_xgboost$SalePrice <- NULL
#test_lm$SalePrice <- NULL

train_xgboost$IsTrain <- NULL
test_xgboost$IsTrain <- NULL
#test_lm <- test_lm[,impVars]
train_xgboost_sales <- train_xgboost$SalePrice
train_xgboost$SalePrice <- NULL

#Use Only Features evaluated as Accepted by Boruta
#impVars
train_xgboost <- train_xgboost[,impVars]
test_xgboost <- test_xgboost[,impVars]

train_xgBoost <- xgb.DMatrix(data.matrix (train_xgboost),label = train_xgboost_sales)
xgmodel <- xgboost(data=train_xgBoost,nrounds=1000)
xgmodel_predict <- predict(xgmodel,data.matrix(test_xgboost))

response_df <- data.frame(Id=test$Id,SalePrice=xgmodel_predict)
summary(response_df)
summary(response_df)
str(response_df)
write.csv(response_df,file="D:/amit/Data Science/Kaggle/House Prices/XGboost_Borutafeatures.csv",row.names=FALSE)
