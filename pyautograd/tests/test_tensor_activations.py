import unittest
from pyautograd import Tensor
from pyautograd.tensor.activations import ReLU, Tanh, Sigmoid, Softmax
import numpy as np


class TestTensorActivation(unittest.TestCase):
    def test_tensor_relu(self):
        t1 = Tensor([[1, 2, 3], [-3, -4, -5], [-7, 8, -9]], requires_grad=True)

        t2 = t1.activation(ReLU)

        assert t2.data.tolist() == [[1, 2, 3], [0, 0, 0], [0, 8, 0]]

        t2.backward()

        assert t1.grad.data.tolist() == [[1, 1, 1], [0, 0, 0], [0, 1, 0]]

    def test_tensor_tanh(self):
        t1 = Tensor(
            [[1.0, 2.0, 3.0], [-3.0, -4.0, -5.0], [-7.0, 8.0, -9.0]], requires_grad=True
        )

        t2 = t1.activation(Tanh)

        np.testing.assert_almost_equal(t2.data, np.tanh(t1.data))

        t2.backward()

        np.testing.assert_almost_equal(t1.grad.data, (1 - (t2 * t2)).data)

    def test_tensor_sigmoid(self):
        t1 = Tensor(
            [[1.0, 2.0, 3.0], [-3.0, -4.0, -5.0], [-7.0, 8.0, -9.0]], requires_grad=True
        )

        t2 = t1.activation(Sigmoid)

        np.testing.assert_almost_equal(t2.data, 1 / (1 + np.exp(-t1.data)))

        t2.backward()

        np.testing.assert_almost_equal(t1.grad.data, (t2 * (1 - t2)).data)

    def test_tensor_softmax(self):
        t1 = Tensor(
            np.array([[1.0, 2.0, 3.0], [-3.0, -4.0, -5.0], [-7.0, 8.0, -9.0]]),
            requires_grad=True,
        )

        t2 = Softmax.forward(t1)
        exp_t = np.exp(t1.data - np.max(t1.data, axis=-1, keepdims=True))
        expected_softmax = exp_t / np.sum(exp_t, axis=-1, keepdims=True)
        np.testing.assert_almost_equal(t2.data, expected_softmax)

        t2.backward()

        grad = np.ones_like(t2.data)

        s = t2.data
        expected_grad = grad - np.sum(grad * s, axis=-1, keepdims=True)
        expected_grad *= s
        np.testing.assert_almost_equal(t1.grad.data, expected_grad)
