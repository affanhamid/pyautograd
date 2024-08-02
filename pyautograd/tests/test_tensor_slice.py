import unittest
from pyautograd import Tensor
import numpy as np


class TestTensorMul(unittest.TestCase):
    def test_simple_mul(self):
        t1 = Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], requires_grad=True)

        t3 = t1[0]

        assert t3.data.tolist() == [1, 2, 3]

        t3.backward(Tensor([-1, -2, -3]))

        assert t1.grad.data.tolist() == [[-1, -2, -3], [0, 0, 0], [0, 0, 0]]
