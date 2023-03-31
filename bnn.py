import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def relu(x):
    return np.maximum(0, x)

class BNN:
    weights = np.empty(0)
    biases = np.empty(0)
    def __init__(self, shape, activation=sigmoid, input=2, output=1):
        self.activation = activation 
        np.append(self.weights, np.random.randn(input, shape[0]))
        for i in range(len(shape)-1):
            np.append(self.weights, np.random.randn(shape[i-1], shape[i]))
        np.append(self.weights, np.random.randn(shape[-1], output))
        self.biases = np.random.randn(len(shape))
    
    def forward(self, x, w, b):
        y = x
        for i in range(len(w)):
            y = np.dot(y, w[i])+b[i]
            y = self.activation(y)
        fin = []
        return y
    

    
