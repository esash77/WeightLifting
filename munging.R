

results_test <- sapply(wl_test, FUN = function(x) !all(is.na(x)))

goodNames <- names(results_test[results_test])
keepNames <- goodNames[-c(1,3,4,5,60)]


wl2 <- wl[,keepNames]
wl3 <- wl2
for ( name in names(wl2) ) {
  temp <- wl2[, name]
  if ( !is.numeric(temp)) {
    coercable <- suppressWarnings(as.numeric(temp))
    mixed <- length(coercable[is.na(coercable)]) > 0 && length(coercable[!is.na(coercable)]) > 0
    print(name); print(mixed)
    if ( mixed ) {
      wl3[, name ] <- coercable
    }
  }
}  
  

# wlt2 <- wl_test[, keepNames]


wl3$classe <- factor(wl$classe)
wl3$user_name <- factor(wl3$user_name)
wl3$new_window <- factor(wl3$new_window)

library(caret)

set.seed(3030)
trainIndex <- createDataPartition(y = wl3$classe, p = 0.6, list = FALSE, times = 1)

wlTrain <- wl3[trainIndex, ]
wlTest <- wl3[-trainIndex, ]

names(wl2)
wl.pr <- prcomp(wl2[, -c(1,2)], scale. = TRUE)

# do a quick random forest to see how many trees are needed
# for error rates to settle down
library(randomForest)
rf <- randomForest(x = wlTrain[,1:55], y = wlTrain$classe,
                   do.trace = 50, importance = TRUE)
rf

# looks like 300 trees will do



set.seed(2020)
rf <- train(x = wlTrain[,1:55], y = wlTrain$classe, method = "rf", 
             trControl = trainControl(method = "oob"),
             allowParallel = TRUE, do.trace = 50, ntree = 250, 
             importance = TRUE, tuneLength = 10)
preds <- predict(rf2, newdata = wlTest[,1:55])
confusionMatrix(preds, wlTest$classe)

rf.imp <- varImp(rf, scale = FALSE, type = 1)
plot(rf.imp, top = 10, main = "Variable-Importance Plot",
     xlab = "Importance (Mean Decrease in Accuracy)")

