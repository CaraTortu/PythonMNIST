import numpy as np

class Layer:
    def __init__(self, size) -> None:
        self.size = size

    def initParams(self, prevLayer) -> None:
        self.weigths = np.random.rand(self.size, prevLayer.size) - 0.5
        self.bias = np.zeros((self.size, 1))
