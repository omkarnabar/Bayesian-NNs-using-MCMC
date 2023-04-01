import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def relu(x):
    return np.maximum(0, x)

class BNN:
    weights = []
    biases = []

    def __init__(self, shape, activation=sigmoid):
        self.activation = activation 
        
        for in_dims, out_dims in zip(shape[:-1], shape[1:]):
            w = np.random.normal(0, 1, (in_dims, out_dims))
            b = np.random.normal(0, 1, (out_dims, ))

            self.weights.append(w)
            self.biases.append(b)
        
    
    def forward(self, x, w, b):
        y = x
        for weight, bias in zip(w, b):
            y = y @ weight + bias
            y = self.activation(y)
        return y
    

    
