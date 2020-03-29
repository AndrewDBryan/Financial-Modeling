#Purpose: To predict a target variable for time n + 12 by training two machine learning models
# (neural net and random forest) on time n to n+11, selecting the best model based on
# test values RMSE, and appending model predictions to an output dataset.

#If only scoring a new test set is desired and a model has already been built,
#then that model can be loaded in and used to score more test cases.

#Import Libraries ########################################################################
library(readr)
library(tidyverse)
library(foreach)
library(foreign)
library(caret)
library(neuralnet)
library(stringr)

#Inputs ###########################################################################

#####Global Parameters (Always Fill In)

#What is your working directory?
setwd('~/Desktop/R Projects')

#Do you want to train new models? ('yes' or 'no')
run <- 'yes'

#####Use Old Model Parameters (Fill in if using model on new test cases)

#Load in Old Model Here
#Example: load(name_of_model.RData), then old_model <- name_of_loaded_model
load('last_neural_net.RData')
old_model <- final_net

#Load in test data here to score with the old chosen model.
test_file <- read_csv('test_data.csv')

#What you want to predict in column form, should be same as what model was built for (change right side)
test_file$dependent_variable <- test_file$forward1eresa

#####New Model Parameters (Fill in if you are training a new model)

#Load main data file here (use read.dta instead if you're loading a stata date file)
data <- read_csv('drewdata.csv')
#last_13 <- sort(unique(data$daynum), decreasing = TRUE)[1:13]
#data <- data %>% filter(daynum %in% last_13) #testing on only last 13 time instances for now
data <- data %>% filter(daynum %% 5 == 0)

#Optional Random Seed (4 digit) for Reproducible Results (take out for random results)
set.seed(1234)

#What you want to predict in column form (change right side)
data$dependent_variable <- data$forward1eresa

#Enter in your predictor variables in string form
independent_variables <- c('chgeresa', 'backeresa', 'rangeeresa', 'pchange', 'pback')

#Which time variable do you want to split test/train by? (change just part after '$')
data$train_test_variable <- data$daynum

#How many time instances do you want to be a part of your training data?
train_size <- 12

#Start ############## Can Run Rest of Code from Here ###############################

raw <- data
data <- data[complete.cases(data),]

#Giant Loop
if (run == 'yes'){

#put predictors in a string together with '+'s' in between for formula
predictors <- paste(as.vector(independent_variables), collapse='+')

#Run Models ######################################################################################

#Initialize Empty Lists and Vectors
formula <- as.formula(paste('dependent_variable ~ ', predictors, sep = ''))
model_types <- c('nnet','ranger')
model <- list('neural net', 'random forest')
model_predictions <- list()
R_squared <- list()
rms_error <- list()
test_predictions <- list()
net_models <- list()
nnet_test_predictions <- list()
nnet_train_predictions <- list()
nnet_total_predictions <- list()
nnet_R_squared <- c()
nnet_train_rmse <- c()
nnet_test_rmse <- c()
forest_model <- list()
forest_test_predictions <- list()
forest_train_predictions <- list()
forest_total_predictions <- list()
forest_R_squared <- c()
forest_train_rmse <- c()
forest_test_rmse <- c()

#Need to have enough data for train and test
time_length <- length(unique(data$train_test_variable)) - train_size

#Rank unique time variable to create partitions for run
key_temp <- data %>% select(train_test_variable) %>% unique()
key_temp$train_test_variable <- as.numeric(key_temp$train_test_variable)
key_temp <- key_temp %>% arrange(train_test_variable)
key <- data.frame(key_temp$train_test_variable, seq(1:length(key_temp$train_test_variable)))
names(key) <- c('train_test_variable', 'time_variable')
data <- data %>% inner_join(key, by = 'train_test_variable')
data <- data %>% arrange(time_variable)

#Tuning Grids (Can be edited but will linearly affect runtime)
neural_net_grid <- expand.grid(.decay = c(0.5, 0.1), 
                               .size = c(3, 4))
random_forest_grid <- expand.grid(mtry = c(round(length(independent_variables)/2), length(independent_variables)-1), 
                                           splitrule = c('variance'), 
                                           min.node.size = c(3,4))
grids <- list(neural_net_grid, random_forest_grid)

#Run iteration for each train:test pair
for (i in 1:time_length){

#Initialize train and test data for run
train_data = data[which(data$time_variable >= i & data$time_variable<= i+(train_size - 1)),]
test_data = data[which(data$time_variable == i+train_size),]
total_data = data[which(data$time_variable >= i & data$time_variable<= i+train_size),]

#Neural Net Model
net_models[[i]] <- caret::train(form = formula, 
                      data = train_data,
                      method = model_types[1],
                      maxit = 100,
                      trace = FALSE,
                      linout = 1,
                      tuneGrid = data.frame(grids[1]),
                      na.action = 'na.omit')

#Make predictions on the test data
nnet_test_predictions[[i]] <- predict(net_models[[i]], test_data)
nnet_train_predictions[[i]] <- predict(net_models[[i]], train_data)
nnet_total_predictions[[i]] <- predict(net_models[[i]], total_data)

#Check
if (length(nnet_test_predictions[[i]]) < nrow(test_data)){
  print(paste0("Model set ", i, " for neural net did not work."))
}

#calculate the R squared value
nnet_R_squared[[i]] <- cor(nnet_test_predictions[[i]][,1], test_data$dependent_variable)^2

#collect the training RMSE
nnet_train_rmse[[i]] <- mean(sqrt((train_data$dependent_variable - nnet_train_predictions[[i]])^2))

#calculate the testing RMSE (used to pick best model)
nnet_test_rmse[[i]] <- mean(sqrt((test_data$dependent_variable - nnet_test_predictions[[i]])^2))

#Random Forest Model
forest_model[[i]] <- caret::train(form = formula, 
                          data = train_data,
                          method = model_types[2],
                          tuneGrid = data.frame(grids[2]),
                          na.action = 'na.omit')

#Make predictions on the test data
forest_test_predictions[[i]] <- predict(forest_model[[i]], test_data)
forest_train_predictions[[i]] <- predict(forest_model[[i]], train_data)
forest_total_predictions[[i]] <- predict(forest_model[[i]], total_data)

#Check
if (length(forest_test_predictions[[i]]) < nrow(test_data)){
  print(paste0("Model set ", i, " for random forest did not work."))
}

#calculate the R squared value
forest_R_squared[[i]] <- cor(forest_test_predictions[[i]], test_data$dependent_variable)^2

#collect the training RMSE
forest_train_rmse[[i]] <- mean(sqrt((train_data$dependent_variable - forest_train_predictions[[i]])^2))

#calculate the testing RMSE (used to pick best model)
forest_test_rmse[[i]] <- mean(sqrt((test_data$dependent_variable - forest_test_predictions[[i]])^2))

print(paste0('Model set ',i,' of ',time_length,' is complete.'))
}

#Take Mean of Data Points Saved Above For Each Model
nnet_final_test_rmse <- mean(unlist(nnet_test_rmse))
forest_final_test_rmse <- mean(unlist(forest_test_rmse))
nnet_final_R_squared <- mean(unlist(nnet_R_squared), na.rm = TRUE)
forest_final_R_squared <- mean(unlist(forest_R_squared), na.rm = TRUE)

#Determine Best Model Based on Root Mean Squared Error
final_net <- net_models[[time_length]]
final_forest <- forest_model[[time_length]]

if (as.numeric(nnet_final_test_rmse) < as.numeric(forest_final_test_rmse)){
  best_model <- model[1]
  predicted_values <- unlist(nnet_test_predictions)
  best_avg_R_squared <- nnet_final_R_squared
  save(final_net, file = 'last_neural_net.RData')
} else{
  best_model <- model[2]
  predicted_values <- unlist(forest_test_predictions)
  best_avg_R_squared <- forest_final_R_squared
  save(final_forest, 'last_forest_model.RData')
}

#Append Data from Best Model to Dataset
test_data_subset <- data[which(data$time_variable > 12),]
train_only_subset <- data[which(data$time_variable <= 12),]
test_data_subset$predictions <- predicted_values
train_only_subset$predictions <- NA
final_data <- rbind(test_data_subset, train_only_subset)
final_data$best_model <- best_model
final_data$model_R_squared <- best_avg_R_squared

#Export Data ###################################################################################
write.csv(as.matrix(final_data), file = "output_data.csv")

} #end of giant loop that trains original models

#Will only work if file looks the same as practice input due to join fields
unscored_data <- anti_join(raw, data, by = c('date', 'time'))
write.csv(as.matrix(unscored_data), file = 'bad_data.csv')
 
# #Score Test Cases Here if Run = 'No'###########################################################
if (run == 'no'){

  #Break out test file into complete and non-complete entries
  clean_test <- test_file[complete.cases(test_file),]
  unclean_test <- test_file[!complete.cases(test_file),]

  #Score the test data with the model loaded in
  predicted_test <- predict(old_model, newdata = clean_test)
  test_R_squared <- cor(predicted_test, clean_test$dependent_variable)

  #Join Predictions Back
  scored_test <- cbind(clean_test, predicted_test, test_R_squared)
  scored_test <- scored_test %>% rename(., predictions = predicted_test)

  #Export the Scored Test Dataset & Unscored Data
  write.csv(as.matrix(scored_test), file = 'scored_test.csv')
  write.csv(as.matrix(unclean_test), file = 'bad_test_data.csv')
}

#END ################################################################################
#If you have any questions about the use of this code or adding to it, please
#email drewbryan@ymail.com.

