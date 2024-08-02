import numpy as np
import unittest
from pyautograd import Tensor
from pyautograd.tensor.base.module import Module
from pyautograd.tensor.losses import MSE
from pyautograd.tensor.optimizers import SGD


class MockModel(Module):
    def __init__(self):
        self.w = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        self.b = Tensor(4.0, requires_grad=True)

    def forward(self, X):
        return X @ self.w + self.b


class TestSGD(unittest.TestCase):
    def test_sgd_optimizer(self):
        model = MockModel()
        optimizer = SGD(model)

        X = Tensor(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
        y = Tensor(np.array([14, 32, 50]))

        batch_size = 2
        learning_rate = 0.001
        epochs = 100

        losses = []

        for _ in range(epochs):
            losses.append(optimizer.step(X, y, learning_rate, MSE, batch_size).data)

        assert losses[0] > 4 * losses[-1]
