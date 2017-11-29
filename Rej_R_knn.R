#load data file
library("data.table")
setwd("C:/Users/Rejpal/Documents/Study/Imperial/Modules/On-going/Machine Learning/Assignments/Homework 1/")
ww_df <- fread("winequality-white.csv")

names = c("fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol","quality")

colnames(ww_df) <- names

View(ww_df)

#Add column for good_wine and assign values
ww_df[,"good_wine"] <- NA

ww_df[,"good_wine"] <- ifelse(ww_df[,"quality"]>="6","1","0")

View(ww_df)


#random df and split into training, validation and test data sets
ww_rand <- ww_df[sample(1:nrow(ww_df)), ]

View(ww_rand)

##normalise dataframes
install.packages("clusterSim")
library(clusterSim)

z_ww_rand <- data.Normalization(ww_rand,type="n1",normalization="column")  
View(z_ww_rand)  

#YourNormalizedDataSet <- as.data.frame(lapply(YourDataSet, normalize))

##Split dataset into training, validation and test datasets
#training dataset
ww_train <- z_ww_rand[1:1959,1:11]
View(ww_train)

#training labels
ww_train_lab <- z_ww_rand[1:1959,13]
View(ww_train_lab)

#validation dataset
ww_val <- z_ww_rand[1960:3430,1:11]
View(ww_val)

#validation labels
ww_val_lab <- z_ww_rand[1960:3430,13]
View(ww_val_lab)


#test dataset
ww_test <- z_ww_rand[3431:4898,1:11]
View(ww_test)

#test labels
ww_test_lab <- z_ww_rand[3431:4898,13]
View(ww_test_lab)


#Build the knn model
library(class)
knn_pred_1 <- knn(train = ww_train, test = ww_val, cl = ww_train_lab, k = 1)
View(knn_pred_1)

#install.packages("gmodels")
library(gmodels)

#Confusion matrix
CrossTable(x = ww_val_lab, y = knn_pred_1, prop.chisq = FALSE)

con_mat <- table(ww_val_lab,knn_pred_1)
con_mat
errorrate <- sum(con_mat[row(con_mat) != col(con_mat)]) / sum(con_mat)
errorrate

#iterate for k = 1 to 80 to find the optimal K solution using validation
error_rate <- vector(length=80)
k_vec <- vector(length=80)

for(i in 1:80){
  #Apply knn with k = i
  predict<-knn(train = ww_train, test = ww_val, cl = ww_train_lab, k = i)
  con_mat <- table(ww_val_lab,predict)
  er <- sum(con_mat[row(con_mat) != col(con_mat)]) / sum(con_mat)
  error_rate[i] <- c(er)
  k_vec[i] <- c(i)
}

error_rate_matrix <- cbind(k_vec,error_rate)
error_rate_matrix <- data.frame(error_rate_matrix)
error_rate_matrix
min(error_rate_matrix$error_rate)

optimal_k <- which.min(error_rate_matrix$error_rate)
optimal_k

#CONFIRMATION - optimal_k model using validation data 
knn_pred_optimal <- knn(train = ww_train, test = ww_val, cl = ww_train_lab, k = optimal_k)
View(knn_pred_optimal)

con_mat <- table(ww_val_lab,knn_pred_optimal)
con_mat
errorrate <- sum(con_mat[row(con_mat) != col(con_mat)]) / sum(con_mat)
accuracy <- 1 -errorrate
accuracy

#install.packages("gmodels")
library(gmodels)

#Confusion matrix
CrossTable(x = ww_val_lab, y = knn_pred_optimal, prop.chisq = FALSE)

#--------------------------------------------------------------------
#optimal_k model using test data 
knn_pred_test <- knn(train = ww_train, test = ww_test, cl = ww_train_lab, k = optimal_k)
View(knn_pred_test)

con_mat <- table(ww_test_lab,knn_pred_test)
con_mat
errorrate <- sum(con_mat[row(con_mat) != col(con_mat)]) / sum(con_mat)
accuracy <- 1 - errorrate
accuracy

#install.packages("gmodels")
library(gmodels)

#Confusion matrix
CrossTable(x = ww_test_lab, y = knn_pred_test, prop.chisq = FALSE)



