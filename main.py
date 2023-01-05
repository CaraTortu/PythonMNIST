from lib.network import NeuralNetwork
import pandas 
import numpy as np

allData = pandas.read_csv("./data/train.csv")

print("[+] Read data")

m, n = allData.shape

allData = np.array(allData)

np.random.shuffle(allData)

print("[+] Shuffled data")

testData  = allData[0:500].T
test_Y    = testData[0]
test_X    = testData[1:] / 255.

trainData = allData[500:].T
train_Y   = trainData[0]
train_X   = trainData[1:] / 255.

_, train_m = train_X.shape

print("[+] Data split. Training neural network")

net = NeuralNetwork([784, 10, 10], train_X, train_Y, 0.01, 10000, m)
net.train()