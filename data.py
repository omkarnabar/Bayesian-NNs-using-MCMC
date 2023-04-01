import numpy as np

def generate_data(n_samples=1000):
    X = []
    Y = []

    for _ in range(n_samples):
        x_i = np.random.uniform(0, 1, 2)
        if x_i[0]<0.5 and x_i[1]>=0.5:
            y_i = 1
        elif x_i[0]>=0.5 and x_i[1]<0.5:
            y_i = 1
        else:
            y_i = 0
        
        X.append(x_i)
        Y.append(y_i)

    X = np.stack(X)
    Y = np.array(Y)
    return X, Y


def acc(model, samples, x, y):
    n_samples = len(samples)
    y_pred = np.zeros(len(y))

    for sample in samples:
        w = sample[0]
        b = sample[1]
        y_pred = y_pred + model.forward(x, w, b).flatten()

    y_pred = y_pred/n_samples
    y_pred = np.round(y_pred)
    return np.sum(y_pred==y)/y.shape[0]