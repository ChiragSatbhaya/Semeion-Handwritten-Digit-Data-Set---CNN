library(data.table)
library(mlbench)
library(mxnet)
library(deepnet)
library(h2o)

# Read handwritten digits data
semeion=read.csv(file="c:/semeion.csv",header=FALSE, sep=" ")
semeion <- semeion[-267]

N <- nrow(semeion)
learn <- sample(1:N, size=0.7*N)
nlearn <- length(learn)
ntest <- N - nlearn

train.x <- t(data.matrix(semeion[learn, 1:256]))
train.y <- semeion[learn, 257:266]
test.x <- t(data.matrix(semeion[-learn, 1:256]))
test.y <- semeion[-learn, 257:266]
train_array <- train.x
test_array <- test.x
dim(train_array) <- c(16, 16, 1, ncol(train.x))
dim(test_array) <- c(16, 16, 1, ncol(test.x))
#train.x <- array(as.numeric(unlist(train.x)), dim=c(796, 256, 16))

x <- train.y
names <- as.character(c(0:9))
colnames <- colnames(x)
setnames(x, old=colnames, new=names)
x <- colnames(x)[apply(x,1,which.max)]
train.y <- as.numeric(x)

z <- test.y
names <- as.character(c(0:9))
colnames <- colnames(z)
setnames(z, old=colnames, new=names)
z <- colnames(z)[apply(z,1,which.max)]
test.y <- as.numeric(z)

# Configure the structure of our network

# Input
data <- mx.symbol.Variable('data')

# first conv
conv1 <- mx.symbol.Convolution(data=data, kernel=c(5,5), num_filter=20)
tanh1 <- mx.symbol.Activation(data=conv1, act_type="tanh")
pool1 <- mx.symbol.Pooling(data=tanh1, pool_type="max", kernel=c(2,2), stride=c(2,2))

# second conv
conv2 <- mx.symbol.Convolution(data=pool1, kernel=c(5,5), num_filter=50)
tanh2 <- mx.symbol.Activation(data=conv2, act_type="tanh") 
pool2 <- mx.symbol.Pooling(data=tanh2, pool_type="max", kernel=c(2,2), stride=c(2,2)) 

# first fullc
flatten <- mx.symbol.Flatten(data=pool2) 
fc1 <- mx.symbol.FullyConnected(data=flatten, num_hidden=500) 
tanh3 <- mx.symbol.Activation(data=fc1, act_type="tanh") 

# second fullc
fc2 <- mx.symbol.FullyConnected(data=tanh3, num_hidden=10) 

# loss
lenet <- mx.symbol.SoftmaxOutput(data=fc2) 

devices <- mx.cpu()

# LeNet (num.round=100, array.batch.size=100, learning.rate=0.05, momentum=0.9)
model1 <- mx.model.FeedForward.create(lenet, X=train_array, y=train.y, num.round=100, 
                                     array.batch.size=100, learning.rate=0.05, momentum=0.9,
                                     eval.metric = mx.metric.accuracy,
                                     epoch.end.callback = mx.callback.log.train.metric(100))

#-------------------------------------------------------------------------------

# Predict labels

predicted <- predict(model1, test_array)
# Assign labels

predicted_labels <- max.col(t(predicted)) - 1

# Get accuracy

sum(diag(table(test.y, predicted_labels)))/478

graph.viz(model1$symbol$as.json())
model1$symbol$as.json()
