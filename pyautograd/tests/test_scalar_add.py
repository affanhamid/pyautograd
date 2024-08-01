import unittest
from pyautograd import Number


class TestTensorAdd(unittest.TestCase):
    def test_simple_add(self):
        t1 = Number(1.0)
        t2 = Number(3.9)

        t3 = t1 + t2

        assert t3.data == 4.9

        t3.backward()

        assert t1.grad == 1
