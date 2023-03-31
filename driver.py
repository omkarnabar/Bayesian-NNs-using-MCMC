import numpy as np 
from bnn import *
import data
from mcmc import *

model = BNN(shape=[4])

trainX, trainY = data.generate_data(5000)
testX, testY = data.generate_data(1000)

w = model.weights
b = model.biases
y = model.forward(trainX, w, b)
print(y[0:5])
##samples = metropolis_hastings(model, w, b, trainX, trainY, n_samples=100, n_burn=10)

##print(f"Train_acc: {data.acc(model, samples, trainX, trainY)}")
##print(f"Test_acc: {data.acc(model, samples, testX, testY)}")


