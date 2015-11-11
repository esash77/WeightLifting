set.seed(3030)
resamp1 <- sample(1:71,size = 71, replace = T)
resampData1 <- m111survey[resamp1,]
tesamp.tr.1 <- tree(sex~fastest+GPA+height+sleep+weight_feel+love_first,
                    data = resampData1)


set.seed(3030)
resamp2 <- sample(1:71,size = 71, replace = T)
resampData2 <- m111survey[resamp2,]
tesamp.tr.2 <- tree(sex~fastest+GPA+height+sleep+weight_feel+love_first, data = resampData2)


set.seed(3030)
resamp3 <- sample(1:71,size = 71, replace = T)
resampData3 <- m111survey[resamp3,]
tesamp.tr.3 <- tree(sex~fastest+GPA+height+sleep+weight_feel+love_first, data = resampData3)