from .number import Number


class ReLU:
    @staticmethod
    def apply(a):

        out = Number(
            0 if a.data < 0 else a.data,
            (a,),
            lambda: ReLU.backward(a, out.data, out.grad),
        )

        return out

    @staticmethod
    def backward(a, output, grad):
        a.grad += (output > 0) * grad

    def __repr__():
        return "ReLU"
