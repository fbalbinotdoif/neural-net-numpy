import numpy as np

def step(z):
    "Step activation function"
    "Returns 1 if z > 0, else 0"
    return 1 if z > 0 else 0

def sigmoid(z):
    "Squash z into (0,1)"
    return 1 / (1 + np.e ** -z)

def sigmoid_derivative(z):
    "Derivate of Sigmoid expressed via sigmoid(z) itself"
    return sigmoid(z) * (1 - sigmoid(z))

def relu(z):
    "ReLu: max(0,z), applied element-wise"
    return np.maximum(0, z)

def relu_derivative(z):
    "Derivative of ReLu: 1 where > 0 , 0 otherwise"
    return (z > 0).astype(float)

def softmax(x):
    "Softmax over x, numerically stable (substructure max before exponentiating)"
    shifted = x - np.max(x)
    exp = np.exp(shifted)
    return exp / np.sum(exp)