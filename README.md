# Python MNIST Neural Network written from scratch!

To run:

```sh
# Install the requirements
python3 -m pip install -r requirements.txt

# Run the Neural Network training process
python3 main.py
```

Feel free to modify this script to fit your needs and change the network size if you wish! In the main file we define the network like such:

```py
net = NeuralNetwork([784, 10, 10], train_X, train_Y, 0.01, 10000, m)
```

The first argument is the network dimensions. The first value is the input layer, the second and third values are the training data. The fourth argument is the training rate, the second to last one is the epochs and the last one is the amount of images we are working with.

To add more layers to the network, you can do it like such:

```py
net = NeuralNetwork([784, 10, 50, 10, 10], train_X, train_Y, 0.01, 10000, m)
```

Here we added another two hidden layers where the first one is 10 neurons big and the second one 50 neurons big.

If you get any weird warnings in the console it mean that the learning rate is too high (I spent hours figuring this out so I have you the hassle)