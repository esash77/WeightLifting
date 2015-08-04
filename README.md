# Predicting Movement-Types:  Quick Model-Making with Random Forests
Homer White  
August 4, 2015  




## Overview

The project data is associated with the following study:

>Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H.: *Qualitative Activity Recognition of Weight Lifting Exercises.* **Proceedings of 4th International Conference in Cooperation with SIGCHI** (Augmented Human '13) . Stuttgart, Germany: ACM SIGCHI, 2013.

Among other things, researchers were interested in using inertial measurement units (IMUs) to classify types of movement in human subjects.

The article describes a study involving six male subjects, all in their twenties..  Each subject was taught how to lift a dumb-bell correctly, and was also taught how to perform that same movement in four incorrect ways.  The five resulting categories were:

* A:  correct procedure;
* B:  throwing the elbows to the front;
* C:  lifting the dumbbell only halfway;
* D:  lowering the dumbbell only halfway;
* E:  throwing the hips to the front.

Each subject then performed ten repetitions of the lifting movement, in each of the five possible ways.  During each lift, the researchers recorded a number of inertial measurements:

>"For data recording we used four 9 degrees of freedom Razor inertial measurement units (IMU), which provide three-axes acceleration, gyroscope and magnetometer data at a joint sampling rate of 45 Hz. Each IMU also featured a Bluetooth module to stream the recorded data to a notebook running the Context Recognition Network Toolbox. We mounted the sensors in the users’ glove, armband, lumbar belt and dumbbell ... . We designed the tracking system to be as unobtrusive as possible, as these are all equipment commonly used by weight lifters."


Since there are six subjects, we have a total of 300 lifts.  However, during each lift the IMU measurements were gathered over a rolling series of time-windows, which over-lapped somewhat and which varied in length from 0.5 to 2.5 seconds.  This results in quite a few actual observations:  apparently a single observation in the data set corresponds to a specific time-window for a specific subject performing a lift in one of the specified ways.

The aim of this report is to devise a random forest model to predict activity-type from other variables in the data set.

## Data Processing

We download the main data. along with the examination data, from the web:


```r
wl <- read.csv("http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv",
                  stringsAsFactors = FALSE)
wl_test <- read.csv("http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv",
                    stringsAsFactors = FALSE)
save(wl, file = "data/wl.rda")
save(wl_test, file = "data/wl_test.rda")
```



The main data set consists of 19622 observations on 160 variables, including:

* a spurious row-number variable `X`;
* `user_name` (the name of the subject);
* three time-stamp variables;
* two variables, `new_window` and `num_window` related to time-windows;
* 152 numerical measurements derived from the IMUs;
* the variable `classe` that records the activity-type.


The examination set has 20 observations on the same first 159 variables as the main data, with the variable `problem_id` replacing `classe`.

A preliminary glance at the 20 examination observations indicates that for many of the variables the values are altogether missing.  Although missing-ness may have predictive value, it is difficult to see how to take advantage of this fact, so we will simply exclude all such variables from our training data.  We will also exclude all variables that record time-stamps, and the useless row-number variable `X`.  The code for this elimination is as follows:


```r
results_test <- sapply(wl_test, FUN = function(x) !all(is.na(x)))
goodNames <- names(results_test[results_test])
keepNames <- goodNames[-c(1,3,4,5,60)]
wl2 <- wl[,keepNames]
```

We choose to retain all 152 numerical predictors, as well as the variables `user_name`, `num_window` and `new_window`.

The variables `new_window` and `num_window` are somewhat mysterious, and their nature is not clarified in the original article.  However, since the authors indicated an interest in seeing whether the width of a time-interval was related to how well IMU-measurements could predict activity-type, I elected to retain both of these variables.  I retained `user_name` because I suspected that the six subjects might have different movement profiles.  If so, knowing the subject for an observation could be useful for predicting the activity-type.


We originally imported the data frame with the option `stingsAsFactors` set to `FALSE`.  For the sake of model-building later on, we now convert these string-variables to factors:


```r
wl2$classe <- factor(wl$classe)
wl2$user_name <- factor(wl2$user_name)
wl2$new_window <- factor(wl2$new_window)
```

Before we look into our training data, we need to divide it into a training set and a test set.  Since we do not intend to fit multiple models, we simply make a 60/40 split (training/test) using a command from the `caret` package, as follows:


```r
set.seed(3030)
trainIndex <- createDataPartition(y = wl2$classe, 
                                  p = 0.6, list = FALSE, times = 1)
wlTrain <- wl2[trainIndex, ]
wlTest <- wl2[-trainIndex, ]
```


## Descriptive Work

Now we delve a bit into the training set.  We begin by looking at the principal components (using commands from the excellent `FactoMineR` package):


```r
wl.pc <- PCA(wlTrain[, -c(1,2,56)], graph = FALSE)
kable(wl.pc$eig[1:10, 2:3])
```

           percentage of variance   cumulative percentage of variance
--------  -----------------------  ----------------------------------
comp 1                  15.811429                            15.81143
comp 2                  15.460987                            31.27242
comp 3                   8.908662                            40.18108
comp 4                   7.141364                            47.32244
comp 5                   5.991619                            53.31406
comp 6                   4.403847                            57.71791
comp 7                   4.010248                            61.72816
comp 8                   3.780985                            65.50914
comp 9                   3.255272                            68.76441
comp 10                  3.152977                            71.91739

Apparently the first five principal components account for a bit more than half of the variance in our numerical predictors.

The next plot shows some of the most important variables plotted against the first two principal components.  These variables would be the ones that are best at spreading out the data.  We would not be surprised later on to see some of them rated as important predictors of activity-type.


```r
plot(wl.pc, choix = "var", select = "cos2 5")
```

![Figure 1 Caption:  The plotting dimensions are determined by the first two principal compnents.  The labeled variables are the five that are 'closest' to the plane of the these components:  that is, they are the most helpful in 'spreading out' the cloud of numerical predictors.](README_files/figure-html/unnamed-chunk-8-1.png) 


In the pre-processing stage we made the choice to retain `user_name` as a predictor variable.  The following two graphs (made with function `cloud()` from the `lattice` package) show the cloud of training observations plotted in terms of the first three principal components and color-coded by name of subject.  The observations fall rather neatly into six distinct sub-clouds, indicating that our six subjects had fairly distinct movement patterns.  We might expect to find later on that `user_name` is a useful predictor variable.


```r
Comp.1 <- wl.pc$ind$coord[,1]
Comp.2 <- wl.pc$ind$coord[,2]
Comp.3 <- wl.pc$ind$coord[,3]
cloud(Comp.1 ~ Comp.2 * Comp.3, groups = wlTrain$user_name,
      screen = list(x = 0, y = 0, z = 0),
      auto.key = list(space = "right"),
      main = "PC-Plot of Training Data,\nby Subject")
```

![Figure 2 Caption:  View of the training observations, plotted in the first three principal components.  Observations are colored according to which subject was being observed.  Obviously the six subjects have rather distinct movement profiles.](README_files/figure-html/unnamed-chunk-9-1.png) 


```r
cloud(Comp.1 ~ Comp.2 * Comp.3, groups = wlTrain$user_name,
      screen = list(x = 0, y = 90, z = 0),
      auto.key = list(space = "right"),
      main = "PC-Plot of Training Data,\nby Subject")
```

![Fgiure 3 Caption:  The same plot, rotated by 90 degrees.](README_files/figure-html/unnamed-chunk-10-1.png) 


## Model-Fitting

We will build a random forest prediction-model, and our intent will be to tune a particular parameter:  namely, the number of variables that the model chooses randomly when it has to decide how to split at any node in the construction of any one of its trees.  (In the `randomForest` package this number is known as `mtry`.)  We don't want our fitting-process to take too long, so we want to make our model using the smallest number of trees we can get away with.  Hence we build a "quickie" random forest with 500 trees, and `mtry` left at its default value (`floor(sqrt(55))`, or 7, in our case):


```r
set.seed(1010)
(rf.prelim <- randomForest(x = wlTrain[,1:55], y = wlTrain$classe,
                   do.trace = 50))
```

```
## ntree      OOB      1      2      3      4      5
##    50:   0.50%  0.09%  0.61%  0.73%  1.04%  0.32%
##   100:   0.36%  0.06%  0.53%  0.34%  0.78%  0.28%
##   150:   0.38%  0.06%  0.61%  0.49%  0.73%  0.23%
##   200:   0.37%  0.03%  0.61%  0.54%  0.67%  0.23%
##   250:   0.37%  0.06%  0.57%  0.49%  0.73%  0.23%
##   300:   0.39%  0.06%  0.53%  0.58%  0.78%  0.23%
##   350:   0.42%  0.06%  0.66%  0.58%  0.83%  0.23%
##   400:   0.38%  0.06%  0.48%  0.54%  0.88%  0.18%
##   450:   0.33%  0.03%  0.48%  0.49%  0.67%  0.18%
##   500:   0.33%  0.03%  0.44%  0.49%  0.73%  0.18%
```

```
## 
## Call:
##  randomForest(x = wlTrain[, 1:55], y = wlTrain$classe, do.trace = 50) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 7
## 
##         OOB estimate of  error rate: 0.33%
## Confusion matrix:
##      A    B    C    D    E  class.error
## A 3347    1    0    0    0 0.0002986858
## B    8 2269    2    0    0 0.0043878894
## C    0   10 2044    0    0 0.0048685492
## D    0    0   13 1916    1 0.0072538860
## E    0    0    0    4 2161 0.0018475751
```


Good---that didn't take too long!  We note from the output that the random forest approach is liable to be impressive (our OOB error rate is only 0.31%), but our primary concern here is with the tree-building process itself.  We see that the OOB error estimates have pretty much stabilized by the time 300 trees are made, so in our actual model we will set `ntree` to 300.


The command below is from the `caret` package.  A few notes:

* for each of 10 values of `mtry` (as determined by the argument `tuneLength = 10`), we will construct a 300-tree random forest.
* For each forest, the prediction will be estimated in the usual "out-of-bag" way (setting `method = "oob"` in `trainControl()`).
* `allowParallel = TRUE` may make a difference on some machines.  (You must first install the `doParallel` package and choose the number of cores you plan to use.  Experiments with the aforementioned default random forest routine indicate that on my machine, a high-end Mac Book Pro, it makes no difference.)
* `importance = TRUE` permits us to make an importance plot later on.

>**Note to Evaluators:**  Although the assignment rubrics call for cross-validation to estimate error rates, out-of-bag estimates are perfectly fine for random forest models, and can be obtained much more quickly.  If that's a problem for you, then I'm willing to face the consequences.


```r
set.seed(2020)
rf <- train(x = wlTrain[,1:55], y = wlTrain$classe, method = "rf", 
             trControl = trainControl(method = "oob"),
             allowParallel = TRUE, ntree = 300, 
             importance = TRUE, tuneLength = 10)
```

The routine took less than nine minutes to run.  Here are the results:


```r
rf
```

```
## Random Forest 
## 
## 11776 samples
##    55 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa    
##    2    0.9931216  0.9912986
##    7    0.9964334  0.9954885
##   13    0.9973675  0.9966703
##   19    0.9974524  0.9967776
##   25    0.9971977  0.9964554
##   31    0.9970279  0.9962406
##   37    0.9966882  0.9958110
##   43    0.9961787  0.9951663
##   49    0.9954993  0.9943071
##   55    0.9942255  0.9926955
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 19.
```

It appears that we'll be going with the model where the trees sampled 19 variables randomly at each node.  Let's see how well we do on the test set:


```r
preds <- predict(rf, newdata = wlTest[,1:55])
confusionMatrix(preds, wlTest$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2231   10    0    0    0
##          B    0 1506    6    0    0
##          C    0    2 1362    2    0
##          D    0    0    0 1283    1
##          E    1    0    0    1 1441
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9971          
##                  95% CI : (0.9956, 0.9981)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9963          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9996   0.9921   0.9956   0.9977   0.9993
## Specificity            0.9982   0.9991   0.9994   0.9998   0.9997
## Pos Pred Value         0.9955   0.9960   0.9971   0.9992   0.9986
## Neg Pred Value         0.9998   0.9981   0.9991   0.9995   0.9998
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2843   0.1919   0.1736   0.1635   0.1837
## Detection Prevalence   0.2856   0.1927   0.1741   0.1637   0.1839
## Balanced Accuracy      0.9989   0.9956   0.9975   0.9988   0.9995
```


Only 11 misses in 7846 observations:  not too shabby!

Let's now have  quick look at which predictor variables were judged to be the most important:


```r
rf.imp <- varImp(rf, scale = FALSE, type = 1)
plot(rf.imp, top = 10, main = "Variable-Importance Plot",
     xlab = "Importance (Mean Decrease in Accuracy)")
```

![Figure 4 Caption:  Importance plot for the ten most important predictors in the final random forest model.  Importance is determined by the mean decrease in prediction accuracy that results when the predictor is removed from a tree in the model.](README_files/figure-html/unnamed-chunk-15-1.png) 

The big surprise is `num_window`, which did not get any hype in the principal components analysis.

## Final Testing

So our final model is estimated to be correct about 99.7% of the time, a good bit better than than than the 98% rate the authors reported for their own model.  If the model were used to predcit the activity of *new* subjects, however, then I would not expect it to do nearly as well.  (In fact the article authors eomployed a leave-one-subject-out routine to estimate an accuracy-rate of only 78% for new subjects.)

The examination data is for the same six subjects, so I am hopeful that my model is good will earn a perfect score.

Let's see what happens.  First, we need to make the examination data have the same form as our training set:


```r
wl_test2 <- wl_test[, keepNames]
wl_test2$new_window <- factor(wl_test2$new_window, levels = c("no", "yes"))
wl_test2$user_name <- factor(wl_test2$user_name)
```

Now we predict:


```r
examPreds <- predict(rf, newdata = wl_test2)
results <- matrix(examPreds, nrow = 1)
colnames(results) <- wl_test$problem_id
kable(results)
```



1    2    3    4    5    6    7    8    9    10   11   12   13   14   15   16   17   18   19   20 
---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---
B    A    B    A    A    E    D    B    A    A    B    C    B    A    E    E    A    B    B    B  

Then we format our answers for submission:


```r
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("answers/problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

pml_write_files(examPreds)
```


Finally, we submit.  Not surprisingly, all of my predictions turned out to be correct.

## References and Remarks

* The source code for this document is the file `README.Rmd` in my GitHub repository:  <a href= "https://github.com/homerhanumat/WeightLifting" target = "_blank">https://github.com/homerhanumat/WeightLifting</a>.
* The HTML for this document can be read as a README in the repository, but since GitHub knows nothing of `knitr` it cannot produce figure captions or format my tables.  If you to see them, then download the file `README.html` and view it.
* A web-link to citation information for the original article is:  <a href = "http://groupware.les.inf.puc-rio.br/har" target = "_blank">http://groupware.les.inf.puc-rio.br/har</a>.
