import numpy as np

def generate_data(n_samples=1000):
    X = []
    Y = []
    for i in range(n_samples):
        x1 = np.random.uniform(0, 1, 1)
        x2 = np.random.uniform(0, 1, 1)
        if x1<0.5 and x2>=0.5:
            y = 1
        elif x1>=0.5 and x2<0.5:
            y = 1
        else:
            y = 0
        X.append([x1, x2])
        Y.append(y)

    return X, Y


def acc(model, samples, x, y):
    n_samples = len(samples)
    n_correct = 0
    y_pred = np.zeros(len(y))
    for i in range(n_samples):
        w = samples[i][0]
        b = samples[i][1]
        y_pred += model.forward(x, w, b)
    y_pred = y_pred/n_samples
    if(y_pred>=0.5):
        y_pred = 1
    else:
        y_pred = 0
    return np.sum(y_pred==y)/len(y)