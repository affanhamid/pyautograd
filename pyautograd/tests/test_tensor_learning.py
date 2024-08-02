import unittest
from pyautograd.tensor import Tensor
from pyautograd.tensor.activations import Tanh
from pyautograd.tensor.base.module import Module
import numpy as np


class TestTensorLearning(unittest.TestCase):
    def test_simple_learn(self):
        x_data = Tensor(np.random.randn(100, 3))
        coef = Tensor(np.array([-1, +3, -2]))
        y_data = x_data @ coef + 5

        w = Tensor(np.random.randn(3), requires_grad=True)
        b = Tensor(np.random.randn(), requires_grad=True)

        batch_size = 32
        learning_rate = 0.01

        loss = 0

        for epoch in range(10):
            epoch_loss = 0.0

            for start in range(0, 100, batch_size):
                end = start + batch_size

                w.zero_grad()
                b.zero_grad()

                inputs = x_data[start:end]

                predicted = inputs @ w + b
                actual = y_data[start:end]
                errors = predicted - actual
                loss = (errors * errors).sum()

                loss.backward()
                epoch_loss += loss.data

                w -= learning_rate * w.grad
                b -= learning_rate * b.grad

                loss = epoch_loss

        assert loss < 1e-4

    def test_fizz_buzz(self):
        def binary_encode(x):
            return [x >> i & 1 for i in range(10)]

        def fizz_buzz_encode(x):
            if x % 15 == 0:
                return [0, 0, 0, 1]
            elif x % 5 == 0:
                return [0, 0, 1, 0]
            elif x % 3 == 0:
                return [0, 1, 0, 0]
            else:
                return [1, 0, 0, 0]

        x_train = Tensor([binary_encode(x) for x in range(101, 1024)])
        y_train = Tensor([fizz_buzz_encode(x) for x in range(101, 1024)])

        class FizzBuzzModel(Module):
            def __init__(self, num_hidden: int = 50) -> None:
                self.w1 = Tensor(np.random.randn(10, num_hidden), requires_grad=True)
                self.b1 = Tensor(np.random.randn(num_hidden), requires_grad=True)

                self.w2 = Tensor(np.random.randn(num_hidden, 4), requires_grad=True)
                self.b2 = Tensor(np.random.randn(4), requires_grad=True)

            def predict(self, inputs: Tensor) -> Tensor:
                x1 = inputs @ self.w1 + self.b1
                x2 = x1.activation(Tanh)
                x3 = x2 @ self.w2 + self.b2

                return x3

        batch_size = 32
        learning_rate = 0.001
        model = FizzBuzzModel()
        first_loss = 0
        last_loss = 1

        starts = np.arange(0, x_train.shape[0], batch_size)
        for epoch in range(100):
            epoch_loss = 0.0

            np.random.shuffle(starts)
            for start in starts:
                end = start + batch_size

                model.zero_grad()

                inputs = x_train[start:end]

                predicted = model.predict(inputs)
                actual = y_train[start:end]
                errors = predicted - actual
                loss = (errors * errors).sum()

                loss.backward()
                epoch_loss += loss.data

                for parameter in model.parameters():
                    parameter -= learning_rate * parameter.grad

            if not first_loss:
                first_loss = np.sum(epoch_loss)

            last_loss = np.sum(epoch_loss)

        assert first_loss > 2 * last_loss
