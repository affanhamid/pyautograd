import unittest
from pyautograd import Tensor
from pyautograd.tensor.losses import MSE, HingeLoss, CrossEntropyLoss
import numpy as np


class TestTensorLoss(unittest.TestCase):
    def test_tensor_mse(self):
        y = Tensor([[1, 4, 5], [-3, -2, -5], [-7, 2, 3]], requires_grad=True)
        p = Tensor([[1, 4, 3], [-3, -2, -5], [-7, 2, 3]], requires_grad=True)

        l = MSE.apply(p, y)

        np.testing.assert_almost_equal(l.data, np.mean(np.square(y.data - p.data)))

        l.backward()

        np.testing.assert_almost_equal(p.grad.data, 2 * (p.data - y.data) / y.data.size)

    def test_cross_entropy_loss(self):
        preds = Tensor(
            np.array([[0.1, 0.9], [0.8, 0.2], [0.4, 0.6]]), requires_grad=True
        )
        y = Tensor(np.array([[0.1, 1], [1, 0.1], [0.1, 1]]), requires_grad=False)
        loss = CrossEntropyLoss.apply(preds, y)

        # Expected loss computation
        expected_losses = -np.sum(y.data * np.log(preds.data), axis=-1)
        expected_loss = np.sum(expected_losses)

        np.testing.assert_almost_equal(loss.data, expected_loss)

    def test_hinge_loss(self):
        preds = Tensor(np.array([[0.8, -0.5], [1.2, -1.5]]), requires_grad=True)
        y = Tensor(np.array([[1, -1], [1, -1]]), requires_grad=False)
        loss = HingeLoss.apply(preds, y)

        # Expected loss computation
        relu_outputs = np.maximum(0, 1 - y.data * preds.data)
        expected_loss = np.sum(relu_outputs) / preds.data.size

        np.testing.assert_almost_equal(loss.data, expected_loss)

        print("Hinge Loss test passed.")
