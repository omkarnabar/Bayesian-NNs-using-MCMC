import numpy as np
import bnn
import math

def logprior(w, b):
    #stddev = 1, mean = 0
    lp = 0
    for weight, bias in zip(w, b):
        lp -= np.sum(weight**2)
        lp -= np.sum(bias**2)
    return lp/2

def loglikelihood(bnn, w, b, x, y):
    #bnn is a BNN object
    #w is a list of weights
    #x is a numpy array of inputs
    #y is a numpy array of outputs
    y_pred  = bnn.forward(x, w, b)
    ll = 0
    for y_true, y_p in zip(y, y_pred):
        ll += y_true*math.log(y_p) + (1-y_true)*math.log(1-y_p)
    return ll

def logposterior(bnn, w, b, x, y):
    return logprior(w, b) + loglikelihood(bnn, w, b, x, y)

def transition(w, b):
    #w is a list of weights
    #b is a list of biases
    #returns a list of weights and a list of biases
    #stddev = 1, mean = 0
    w_new = []
    b_new = []

    for wi, bi in zip(w, b):
        wn = wi + np.random.normal(0, 1, wi.shape)
        bn = bi + np.random.normal(0, 1, bi.shape)
        w_new.append(wn)
        b_new.append(bn)

    return w_new, b_new


def metropolis_hastings(bnn, w, b, x, y, n_samples=100, n_burn=10):
    accepted = 0
    rejected = 0
    samples = []
    while accepted <= n_samples + n_burn:
        w_new, b_new = transition(w, b)

        prob = min(0, (logprior(w_new, b_new) + loglikelihood(bnn, w_new, b_new, x, y)) - (logprior(w, b) + loglikelihood(bnn, w, b, x, y)))

        r = np.random.uniform(0, 1, 1)
        r = math.log(r)

        if r < prob:
            w = w_new
            b = b_new
            accepted += 1
            print("Accepted: ", accepted)
            samples.append((w, b))
        else:
            rejected += 1
            print("Rejected: ", rejected)

    return samples[n_burn:]


