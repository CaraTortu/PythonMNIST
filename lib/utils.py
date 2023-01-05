import numpy as np

def ReLU(Z):
    return np.maximum(Z, 0)

def dReLU(Z):
    return Z > 0

def softmax(Z: np.ndarray):
    exp = np.exp(Z)
    exp /= sum(exp, exp.size)
    return exp

def OneHotEncoding(Y: np.ndarray):
    hot_Y = np.zeros((Y.size, Y.max() + 1)) # Initalises array with zeros with correct dimensions
    hot_Y[np.arange(Y.size), Y] = 1
    return hot_Y.T