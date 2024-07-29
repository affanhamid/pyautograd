from .number import Number


class Add:
    @staticmethod
    def apply(a, b):
        b = b if isinstance(b, Number) else Number(b)

        out = Number(a.data + b.data, (a, b), lambda: Add.backward(out.grad, a, b))

        return out

    @staticmethod
    def backward(grad, a, b):
        a.grad += grad
        b.grad += grad


class Multiply:
    @staticmethod
    def apply(a, b):
        b = b if isinstance(b, Number) else Number(b)

        out = Number(a.data * b.data, (a, b), lambda: Multiply.backward(out.grad, a, b))

        return out

    @staticmethod
    def backward(grad, a, b):
        a.grad += b.data * grad
        b.grad += a.data * grad


class Power:
    @staticmethod
    def apply(a, b):
        assert isinstance(b, (int, float)), "only supporting int/float powers for now"

        out = Number(a.data**b, (a,), lambda: Power.backward(out.grad, a, b))

        return out

    @staticmethod
    def backward(grad, a, b):
        a.grad += (b * a.data ** (b - 1)) * grad
