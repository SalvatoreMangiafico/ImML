#' @title Tree volume models based on height and diameter
#'
#' @description Decision tree, random forest, support vector machine,
#'              and linear models for fitting tree volume to
#'              height and diameter.
#' 
#' @param data   The data frame to use.
#'               Must contain the numeric variables
#'               \code{Volume}, \code{Height}, and \code{Diameter}.
#' @param plotit If \code{TRUE}, produces a plot of predicted values vs.
#'               observed values.
#' @param setseed If not \code{NULL}, is passed to \code{set.seed} 
#'                 for the analysis.
#' @param verbose If \code{TRUE}, prints the output of each fitted model object.
#' @param ... Additional arguments, currently not used.
#'             
#' @details Calculates mean absolute error, root mean square error,
#'          root relative squared error, and prediction error rate
#'          for train and test partitions of a data frame
#'          using decision tree, random forest, support vector machine,
#'          and linear models for fitting tree volume to
#'          height and diameter.
#'                      
#' @author M. Iqbal Jeelani, \email{jeelani.miqbal@gmail.com},
#'         Salvatore Mangiafico, \email{mangiafico@njaes.rutgers.edu}
#' 
#' @references Jeelani, M.I., Tabassum, A., Rather, K and Gul,M.2023. 
#'             Neural Network Modeling of Height Diameter Relationships 
#'             for Himalayan Pine through Back Propagation Approach. 
#'             Journal of The Indian Society of Agricultural Statistics. 
#'             76(3): 169–178.
#' 
#' @concept Tree volume
#' @concept Decision tree
#' @concept Random forest
#' @concept Support vector machine
#' 
#' @return A data frame consisting of mean absolute error, 
#'         root mean square error,
#'         root relative squared error, and prediction error rate
#'         for train and test partitions
#'         using decision tree, random forest, support vector machine,
#'          and linear model.
#'         
#' @note   The data frame must contain the numeric variables
#'         \code{Volume}, \code{Height}, and \code{Diameter}.
#'         \code{Volume} is used as the dependent variable.
#'         
#'         The gray line in the plot is a 1:1 line.   
#'          
#' @examples
#' data(EastCirclePine)
#' ImML(EastCirclePine, plotit=FALSE, setseed=123)
#' 
#' @importFrom caret createDataPartition MAE RMSE
#' @importFrom dplyr bind_rows
#' @importFrom stats lm predict
#' @importFrom rpart rpart
#' @importFrom randomForest randomForest
#' @importFrom e1071 svm
#' @importFrom ggplot2 ggplot
#' 
#' @export

ImML <- function(data, plotit=TRUE, setseed=NULL, verbose=FALSE, ...) {
  
  # Split data into training and testing sets
  if(!is.null(setseed)){set.seed(setseed)}
  trainIndex <- createDataPartition(data$Volume, p = 0.7, list = FALSE)
  train <- data[trainIndex, ]
  test <- data[-trainIndex, ]
  
  # Linear Regression model
  lrModel <- lm(Volume ~ Diameter + Height, data = train)
  if(verbose){
    cat("\n")
    cat("###########################################################", "\n")
    cat("#############    Linear Regression - Train    #############", "\n")
    print(lrModel)
    }
    
  # Decision Tree model
  dtModel <- rpart(Volume ~ Diameter + Height, data = train)
  if(verbose){
    cat("\n")
    cat("###########################################################", "\n")
    cat("#############      Decision Tree - Train      #############", "\n")
    print(dtModel)
    }
  
  # Random Forest model
  rfModel <- randomForest(Volume ~ Diameter + Height, data = train)
  if(verbose){
    cat("\n")
    cat("##########################################################", "\n")
    cat("#############      Random Forest - Train     #############", "\n")
    print(rfModel)
    }
  
  # SVM model
  svmModel <- svm(Volume ~ Diameter + Height, data = train)
  if(verbose){
    cat("\n")
    cat("##########################################################", "\n")
    cat("############# Support Vector Machine - Train #############", "\n")
    print(svmModel)
    }
  
  # Linear Regression model
  lrModel <- lm(Volume ~ Diameter + Height, data = test)
  if(verbose){
    cat("\n")
    cat("##########################################################", "\n")
    cat("#############    Linear Regression - Test    #############", "\n")
    print(lrModel)
    }
  
  # Decision Tree model
  dtModel <- rpart(Volume ~ Diameter + Height, data = test)
  if(verbose){
    cat("\n")
    cat("##########################################################", "\n")
    cat("#############      Decision Tree - Test      #############", "\n")
    print(dtModel)
    }
  
  # Random Forest model
  rfModel <- randomForest(Volume ~ Diameter + Height, data = test)
  if(verbose){
    cat("\n")
    cat("##########################################################", "\n")
    cat("#############      Random Forest - Test      #############", "\n")
    print(rfModel)
    }
  
  # SVM model
  svmModel <- svm(Volume ~ Diameter + Height, data = test)
  if(verbose){
    cat("\n")
    cat("##########################################################", "\n")
    cat("############# Support Vector Machine - Test  #############", "\n")
    print(svmModel)
    cat("##########################################################", "\n")
    cat("\n")
    }
  
  # Make predictions on the test set
  lrTestPred <- predict(lrModel, newdata = test)
  dtTestPred <- predict(dtModel, newdata = test)
  rfTestPred <- predict(rfModel, newdata = test)
  svmTestPred <- predict(svmModel, newdata = test)
  
  if(plotit){
  
  # Create a data frame for plotting
  plotData <- data.frame(
    x = test$Volume,
    yLR = lrTestPred,
    yDT = dtTestPred,
    yRF = rfTestPred,
    ySVM = svmTestPred
  )
  
  # Plot the predicted values
  
  Plot = ggplot(plotData, aes(x = x)) +
    geom_point(aes(y = yLR, color = "Linear Regression")) +
    geom_point(aes(y = yDT, color = "Decision Tree")) +
    geom_point(aes(y = yRF, color = "Random Forest")) +
    geom_point(aes(y = ySVM, color = "Support Vector Machine")) +
    geom_abline(slope=1, intercept=0, color="darkgray") +
    #geom_smooth(aes(y = yLR), method = "lm", formula = y ~ x, se = FALSE, color = "blue") +
    #geom_smooth(aes(y = yDT), method = "lm", formula = y ~ x, se = FALSE, color = "green") +
    #geom_smooth(aes(y = yRF), method = "lm", formula = y ~ x, se = FALSE, color = "purple") +
    #geom_smooth(aes(y = ySVM), method = "lm", formula = y ~ x, se = FALSE, color = "red")  +
    labs(
      title = "Predicted Values for LR, SVM, RF, and DT Models\n",
      x = "\nTrue Volume",
      y = "Predicted Volume\n",
      color = "Model"
    ) +
    theme_bw()

    print(Plot)
    
  }  ### End plotit call
  
  # Evaluate models on the training data
  trainLR <- data.frame(
    Data = "Train",
    Model = "LR",
    MAE = caret::MAE(predict(lrModel, newdata = train), train$Volume),
    RMSE = caret::RMSE(predict(lrModel, newdata = train), train$Volume),
    RRSE = sqrt(mean((predict(lrModel, newdata = train) - train$Volume)^2) / mean(train$Volume^2)),
    PER = mean(abs(predict(lrModel, newdata = train) - train$Volume) / train$Volume) * 100
  )
  
  trainDT <- data.frame(
    Data = "Train",
    Model = "DT",
    MAE = caret::MAE(predict(dtModel, newdata = train), train$Volume),
    RMSE = caret::RMSE(predict(dtModel, newdata = train), train$Volume),
    RRSE = sqrt(mean((predict(dtModel, newdata = train) - train$Volume)^2) / mean(train$Volume^2)),
    PER = mean(abs(predict(dtModel, newdata = train) - train$Volume) / train$Volume) * 100
  )
  
  trainRF <- data.frame(
    Data = "Train",
    Model = "RF",
    MAE = caret::MAE(predict(rfModel, newdata = train), train$Volume),
    RMSE = caret::RMSE(predict(rfModel, newdata = train), train$Volume),
    RRSE = sqrt(mean((predict(rfModel, newdata = train) - train$Volume)^2) / mean(train$Volume^2)),
    PER = mean(abs(predict(rfModel, newdata = train) - train$Volume) / train$Volume) * 100
  )
  
  trainSVM <- data.frame(
    Data = "Train",
    Model = "SVM",
    MAE = caret::MAE(predict(svmModel, newdata = train), train$Volume),
    RMSE = caret::RMSE(predict(svmModel, newdata = train), train$Volume),
    RRSE = sqrt(mean((predict(svmModel, newdata = train) - train$Volume)^2) / mean(train$Volume^2)),
    PER = mean(abs(predict(svmModel, newdata = train) - train$Volume) / train$Volume) * 100
  )
  
  # Evaluate models on the testing data
  testLR <- data.frame(
    Data = "Test",
    Model = "LR",
    MAE = caret::MAE(predict(lrModel, newdata = test), test$Volume),
    RMSE = caret::RMSE(predict(lrModel, newdata = test), test$Volume),
    RRSE = sqrt(mean((predict(lrModel, newdata = test) - test$Volume)^2) / mean(test$Volume^2)),
    PER = mean(abs(predict(lrModel, newdata = test) - test$Volume) / test$Volume) * 100
  )
  
  testDT <- data.frame(
    Data = "Test",
    Model = "DT",
    MAE = caret::MAE(predict(dtModel, newdata = test), test$Volume),
    RMSE = caret::RMSE(predict(dtModel, newdata = test), test$Volume),
    RRSE = sqrt(mean((predict(dtModel, newdata = test) - test$Volume)^2) / mean(test$Volume^2)),
    PER = mean(abs(predict(dtModel, newdata = test) - test$Volume) / test$Volume) * 100
  )
  
  testRF <- data.frame(
    Data = "Test",
    Model = "RF",
    MAE = caret::MAE(predict(rfModel, newdata = test), test$Volume),
    RMSE = caret::RMSE(predict(rfModel, newdata = test), test$Volume),
    RRSE = sqrt(mean((predict(rfModel, newdata = test) - test$Volume)^2) / mean(test$Volume^2)),
    PER = mean(abs(predict(rfModel, newdata = test) - test$Volume) / test$Volume) * 100
  )
  
  testSVM <- data.frame(
    Data = "Test",
    Model = "SVM",
    MAE = caret::MAE(predict(svmModel, newdata = test), test$Volume),
    RMSE = caret::RMSE(predict(svmModel, newdata = test), test$Volume),
    RRSE = sqrt(mean((predict(svmModel, newdata = test) - test$Volume)^2) / mean(test$Volume^2)),
    PER = mean(abs(predict(svmModel, newdata = test) - test$Volume) / test$Volume) * 100
  )
  
  # Combine results into one dataframe
  results <- bind_rows(trainLR, trainDT, trainRF, trainSVM, testLR, testDT, testRF, testSVM)
  
  return(results)
}
