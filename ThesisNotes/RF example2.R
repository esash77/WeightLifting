library(tigerstats)
dfNames <- names(m111survey)
predNames <- dfNames[dfNames != "sex"]
m111s2 <- m111survey[complete.cases(m111survey),]
df.pred <- m111s2[,predNames]

numberTrain <- floor(2/3*68)
inTrain <- c(rep(TRUE,45), rep(FALSE,68-45))
inTrain <- sample(inTrain, size = 68)
m111.train <- df.pred[inTrain,]
m111.test <- df.pred[!inTrain,]
y.train <- m111s2$sex[inTrain]
y.test <- m111s2$sex[!inTrain]

library(randomForest)
rf.m111 <- randomForest(x = m111.train, y = y.train, xtest = m111.test, ytest = y.test, do.trace = 50)
rf.m111 

importance(rf.m111)

varImpPlot(rf.m111)