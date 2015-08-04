table(wl_test$user_name)
table(wl$user_name)

useful <- function(x) {
  allNA <- all(is.na(x))
  if (!allNA) {
    oneValue <- length(unique(x)) == 1
  }
  useful <- !allNA && !oneValue
  return(useful)
}

results_test <- sapply(wl_test, FUN = function(x) !all(is.na(x)))
results_train <- sapply(wl, FUN = useful)
diff <- results_train[results_train != results_test]
diff

goodNames <- names(results_test[results_test])
keepNames <- goodNames[-c(1,3,4,5,60)]
keepNames


library(lattice)
library(tigerstats)


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
  

wlt2 <- wl_test[, keepNames]

library(randomForest)

wl3$classe <- factor(wl$classe)
wl3$user_name <- factor(wl3$user_name)
wl3$new_window <- factor(wl3$new_window)
trainNumber <- ceiling(nrow(wl3)*0.60)
testNumber <- nrow(wl3) - trainNumber
temp <- c(rep(TRUE, trainNumber), rep(FALSE, testNumber))
selected <- sample(temp, replace = FALSE)
wlTrain <- subset(wl3, selected)
wlTest <- subset(wl3, !selected)

rf <- randomForest(x = wlTrain[,1:55], y = wlTrain$classe,
                   xtest = wlTest[, 1:55], ytest = wlTest$classe, 
                   do.trace = 50, importance = TRUE)
rf$confusion

importance(rf)
varImpPlot(rf, n.var = 10)

wlTrainSS <- split(wlTrain, f = wlTrain$user_name)
wlTestSS <- split(wlTest, f = wlTest$user_name)

forests <- list()
for ( i in 1:6 ) {
  print(unique(wlTrainSS[[i]]$user_name))
  forests[[i]] <- randomForest(x = wlTrainSS[[i]][,1:55], 
                               y = wlTrainSS[[i]]$classe,
                               xtest = wlTestSS[[i]][, 1:55], 
                               ytest = wlTestSS[[i]]$classe, 
                               do.trace = 50)
}

confusion <- forests[[1]]$confusion + forests[[2]]$confusion + 
             forests[[3]]$confusion + forests[[4]]$confusion +
             forests[[5]]$confusion + forests[[6]]$confusion
confusion
table(wl_test$user_name)
