import unittest
from pyautograd import Tensor
import numpy as np


class TestTensorMul(unittest.TestCase):
    def test_simple_log(self):
        t1 = Tensor([3.4, 5.9, 0.1, 2.3, 23.5, 4.3], requires_grad=True)

        t2 = t1.log()

        np.testing.assert_almost_equal(t2.data, np.log(t1.data))

        # t3.backward(Tensor([-1, -2, -3]))

        # assert t1.grad.data.tolist() == [-4, -10, -18]
        # assert t2.grad.data.tolist() == [-1, -4, -9]
