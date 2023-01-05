import numpy as np
import matplotlib.pyplot as plt

np.seterr(all='raise')

def ReLU(Z):
    return np.maximum(Z, 0)

def dReLU(Z):
    return Z > 0

def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

def dsigmoid(Z):
    return np.exp(-Z)/(np.exp(-Z)+1)**2

def softmax(Z: np.ndarray):
    try:
        exp = np.exp(Z)
        exp /= sum(exp, exp.size)
    except:
        print("[-] Learning rate is too high. Please lower it")
        exit()

    return exp

def OneHotEncoding(Y: np.ndarray):
    hot_Y = np.zeros((Y.size, Y.max() + 1)) # Initalises array with zeros with correct dimensions
    hot_Y[np.arange(Y.size), Y] = 1
    return hot_Y.T