import unittest
from pyautograd import Number
import numpy as np


class TestScalarMul(unittest.TestCase):
    def test_simple_mul(self):
        t1 = Number(4.5)
        t2 = Number(2.8)

        t3 = t1 * t2

        assert np.isclose(t3.data, 12.6)

        t3.backward()

        assert t1.grad == 2.8
        assert t2.grad == 4.5
