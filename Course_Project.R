setwd("C:/users/acer/desktop/Coursera/PracticalMachineLearning")

library(caret)
library(randomForest)

#Load the Data
df = read.csv("pml-training.csv", header=TRUE, 
              na.strings=c("NA", ""), stringsAsFactors=FALSE)

#Data preparation:
  #1. Remove columns with more than 90% NAs
  #2. Remove non-numeric columns - user name, timestamps, new window
  #3. Cast y variable as a factor
NA_cols = colSums(is.na(df)) > 0.9*nrow(df)
df2 = df[, !NA_cols] 
df2 = subset(df2, select=-c(X, 
                            cvtd_timestamp, 
                            user_name, 
                            raw_timestamp_part_1,
                            raw_timestamp_part_2, 
                            new_window, 
                            num_window))
df2$classe = factor(df2$classe)

#Data preparation continued:
  #4. Remove variables that are highly correlated to decrease dimensionality

y_col_idx = grep("classe", names(df2))
#remove correlated variables
corr = cor(df2[,-y_col_idx])
highlyCorrCols = findCorrelation(corr, cutoff = 0.75) 

df3 = df2[, -highlyCorrCols]
y_col_idx = grep("classe", names(df3))
df3_X = df3[,-y_col_idx]
df3_y = df3$classe

#Model selection and development 
  #1. Mulitple iterations of running k-fold (k=5) cross-validation
  #2. Algorithm finally used randomForest

iterations = 10
cv_error = array(0, dim = c(iterations))

for (i in 1:iterations)
{
  set.seed(i)
  n_folds = 5 #k fold validation
  folds = createFolds(df3_y, k=n_folds, list=FALSE)
  
  #declare variables to store accuracy
  train_accuracy = array(0, dim = c(n_folds))
  test_accuracy = array(0, dim = c(n_folds))
  oos_error = array(0, dim = c(n_folds))
  
  for (j in 1:n_folds)
  {
    training_X = df3_X[folds!=j, ]
    training_y = df3_y[folds!=j]
    
    testing_X = df3_X[folds==j, ]  
    testing_y = df3_y[folds==j]  
    fit = randomForest(x=training_X, y=training_y, ntree=20) 

    #high degree of prediction accuracy has been observed even with a small value of ntree
    #smaller ntree helps decrease run time and allow us to run more iterations
    
    train_accuracy[j] = sum(training_y == 
                              predict(fit, training_X)) / nrow(training_X)
    test_accuracy[j] = sum(testing_y == 
                             predict(fit, testing_X))  / nrow(testing_X)    
    oos_error[j] = sum(testing_y != 
                             predict(fit, testing_X))  / nrow(testing_X)        
  }  
  cv_error[i] = mean(oos_error) 
}
print(cv_error)
print(mean(cv_error)) #final cross-validation error
print(fit)

