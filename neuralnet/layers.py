import numpy as np
from neuralnet.activations import step

def perceptron_2d(x1, x2, w1, w2, b):
    z = x1*w1 + x2*w2 + b
    return step(z)

def train_perceptron(X, y, lr=0.1, epochs=10):
    w = np.zeros(X.shape[1])
    b = 0.0

    for epoch in range(epochs):
        for xi, yi in zip(X, y):
            z = np.dot(w, xi) + b
            y_prediction = step(z)
            error = yi - y_prediction
            w = w + lr * error * xi
            b = b + lr * error
    return w, b