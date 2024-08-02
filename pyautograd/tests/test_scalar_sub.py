import unittest
from pyautograd import Number
import numpy as np


class TestScalarSub(unittest.TestCase):
    def test_simple_sub(self):
        t1 = Number(8.4)
        t2 = Number(2.1)

        t3 = t1 - t2

        assert np.isclose(t3.data, 6.3)

        t3.backward()

        assert t1.grad == 1
        assert t2.grad == -1
