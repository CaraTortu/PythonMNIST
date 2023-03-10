from lib.layer import Layer
from lib.utils import *

class NeuralNetwork:
    def __init__(self, layerSizes: list, train_X, train_Y, alpha, m) -> None:
        # Define layers
        self.inputLayer = Layer(layerSizes[0])
        self.layers = [Layer(i) for i in layerSizes[1:]]

        # Get our data
        self.train_X = train_X
        self.train_Y = train_Y
        
        # Variables for training
        self.alpha = alpha
        self.m = m

        # Initialise layer values
        self.initialise()

        self.values = []

    def initialise(self) -> None:
        self.layers[0].initParams(self.inputLayer)

        #Initialise hidden layers
        for (layer, nextLayer) in [(self.layers[i], self.layers[i+1]) for i in range(len(self.layers)-1)]:
            nextLayer.initParams(layer)

        # Initialise output layer
        self.layers[-1].initParams(self.layers[-1])

    def train(self) -> None:
        i = 0

        while True:
            self.forwardProp()
            self.backwardsProp()

            if i % 10 == 0:
                acc = np.sum(np.argmax(self.layers[-1].A, 0) == self.train_Y) / self.train_Y.size

                print(f"Iter: {i} | acc: {acc}")
                self.values.append(acc)

            i+=1

    def forwardProp(self):
        totalValue = self.train_X

        for layer in self.layers[:-1]:
            layer.Z = layer.weigths.dot(totalValue) + layer.bias
            layer.A = sigmoid(layer.Z)
            totalValue = layer.A
            
        self.layers[-1].A = softmax(self.layers[-1].weigths.dot(totalValue) + self.layers[-1].bias)

    def backwardsProp(self):
        oneOverM = 1 / self.m

        dZoutput = self.layers[-1].A - OneHotEncoding(self.train_Y)
        self.layers[-1].weigths -= self.alpha * (oneOverM * dZoutput.dot(self.layers[-2].A.T))
        self.layers[-1].bias    -= self.alpha * (oneOverM * np.sum(dZoutput))

        previous = self.layers[-1]

        for i, layer in list(enumerate(self.layers))[:-1][::-1]:
            dZoutput = previous.weigths.T.dot(dZoutput) * dsigmoid(layer.Z)

            todot = self.train_X.T if i == 0 else self.layers[i-1].A.T

            layer.weigths -= self.alpha * (oneOverM * dZoutput.dot(todot))
            layer.bias -= self.alpha * (oneOverM * np.sum(dZoutput))
            previous = layer
