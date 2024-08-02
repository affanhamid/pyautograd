import unittest
from pyautograd import Tensor


class TestTensorMatmul(unittest.TestCase):
    def test_simple_matmul(self):
        t1 = Tensor([[1, 2], [3, 4], [5, 6]], requires_grad=True)
        t2 = Tensor([[10], [20]], requires_grad=True)

        t3 = t1 @ t2

        assert t3.data.tolist() == [[50], [110], [170]]

        t3.backward(Tensor([[-1], [-2], [-3]]))

        assert t1.grad.data.tolist() == [[-10, -20], [-20, -40], [-30, -60]]
        assert t2.grad.data.tolist() == [[-22], [-28]]
