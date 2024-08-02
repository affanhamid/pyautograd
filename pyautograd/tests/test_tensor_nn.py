import unittest
from pyautograd.tensor.nn import Layer, Network
from pyautograd.tensor import Tensor
from pyautograd.tensor.activations import ReLU
from pyautograd.tensor.optimizers import SGD
from pyautograd.tensor.losses import MSE
import numpy as np


class TestTensorNN(unittest.TestCase):

    def test_layer(self):

        layer = Layer(5, 2, ReLU)

        # Sample input data
        x = Tensor(np.random.randn(3, 5))

        # Perform forward pass
        output = layer.forward(x)

        # Check output shape
        assert output.data.shape == (3, 2)

        # Check weight and bias shapes
        assert layer.w.data.shape == (5, 2)
        assert layer.b.data.shape == (2,)

    def test_layer_train(self):
        input_size = 3
        batches = 5
        layer_size = 1
        model = Layer(input_size, layer_size, ReLU)
        optimizer = SGD(model)

        X = Tensor(np.random.randn(batches, input_size))
        y = Tensor(np.array([14, 12, 45, 2, 27]))

        batch_size = 2
        learning_rate = 0.01
        epochs = 1000

        losses = []

        for _ in range(epochs):
            losses.append(optimizer.step(X, y, learning_rate, MSE, batch_size).data)

        assert losses[0] > 1.5 * losses[-1]

    def test_network_forward(self):
        input_size = 3
        batches = 5

        network = Network([Layer(input_size, 5, ReLU), Layer(5, 5, ReLU), Layer(5, 1)])

        # Sample input data
        x = Tensor(np.random.randn(batches, input_size))

        # Perform forward pass
        output = network.forward(x)

        # Check output shape
        assert output.data.shape == (5, 1)

        output.backward()

    def test_train(self):
        input_size = 3
        batches = 5
        model = Network([Layer(input_size, 5, ReLU), Layer(5, 5, ReLU), Layer(5, 1)])
        optimizer = SGD(model)

        X = Tensor(np.random.randn(batches, input_size))
        y = Tensor(np.array([14, 32, 50, 32, 12]))

        batch_size = 2
        learning_rate = 0.001
        epochs = 100

        losses = []

        for _ in range(epochs):
            losses.append(optimizer.step(X, y, learning_rate, MSE, batch_size).data)
            print(losses[-1])

        assert losses[0] > 3 * losses[-1]
