import numpy as np
import bnn
import math

def logprior(w, b):
    #stddev = 1, mean = 0
    lp = 0
    for i in range(len(w)):
        lp -= np.sum(w[i]**2)
        lp -= b[i]**2
    return lp/2

def loglikelihood(bnn, w, b, x, y):
    #bnn is a BNN object
    #w is a list of weights
    #x is a numpy array of inputs
    #y is a numpy array of outputs
    y_pred  = bnn.forward(x, w, b)
    ll = 0
    for y_true, y_p in zip(y, y_pred):
        ll -= y_true*math.log(y_p) + (1-y_true)*math.log(1-y_p)
    return ll

def logposterior(bnn, w, b, x, y):
    return logprior(w, b) + loglikelihood(bnn, w, b, x, y)

def transition(w, b):
    #w is a list of weights
    #b is a list of biases
    #returns a list of weights and a list of biases
    #stddev = 1, mean = 0
    w_new = np.zeros(w.shape)
    b_new = np.zeros(b.shape)

    for i in range(len(w)):
        w_new[i] = w[i] + np.random.randn(w[i].shape)
        b_new = b[i] + np.random.randn(b[i].shape)

    return w_new, b_new


def metropolis_hastings(bnn, w, b, x, y, n_samples=100, n_burn=10):
    accepted = 0
    samples = []
    while accepted <= n_samples + n_burn:
        w_new, b_new = transition(w, b)
        logposterior_old = logposterior(bnn, w, b, x, y)
        logposterior_new = logposterior(bnn, w_new, b_new, x, y)
        if logposterior_new > logposterior_old:
            w = w_new
            b = b_new
            accepted += 1
            samples.append((w, b))
        else: 
            r = np.random.uniform(0, 1, 1)
            r = np.log(r)
            if r < np.exp(logposterior_new - logposterior_old):
                w = w_new
                b = b_new
                accepted += 1
                samples.append((w, b))

    return samples[n_burn:]


