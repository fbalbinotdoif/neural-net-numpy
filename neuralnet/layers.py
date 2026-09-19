import numpy as np
from neuralnet.activations import step

def perceptron(x, w, b):
    "Weighted sum z = w·x + b (pre-activation, before applying any activation function)"
    z = np.dot(w, x) + b
    return z

def perceptron_2d(x1, x2, w1, w2, b):
    "Class (0 or 1) of the point (x1, x2) for a 2-input perceptron, using the step activation"
    z = x1*w1 + x2*w2 + b
    return step(z)

def train_perceptron(X, y, lr=0.1, epochs=10):
    "Train a perceptron with the perceptron learning rule; returns (w, b)"
    w = np.zeros(X.shape[1])
    b = 0.0

    for epoch in range(epochs):
        for xi, yi in zip(X, y):
            z = perceptron(xi, w, b)
            y_prediction = step(z)
            error = yi - y_prediction
            w = w + lr * error * xi
            b = b + lr * error
    return w, b