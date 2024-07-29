class Number:
    """stores a single scalar value and its gradient"""

    def __init__(self, data, _children=(), _backward=lambda: None):
        self.data = data
        self.grad = 0
        self._backward = _backward
        self._children = set(_children)

    def __add__(self, other):
        from .operations import Add

        return Add.apply(self, other)

    def __mul__(self, other):
        from .operations import Multiply

        return Multiply.apply(self, other)

    def __pow__(self, other):
        from .operations import Power

        return Power.apply(self, other)

    def activation(self, func):
        if func == None:
            return self
        return func.apply(self)

    def topological_sort(self):
        topo_list = []
        visited = set()

        def build_topo_list(n):
            if n not in visited:
                visited.add(n)
                for child in n._children:
                    build_topo_list(child)
                topo_list.append(n)

        build_topo_list(self)

        return topo_list

    def backward(self):
        topo_list = self.topological_sort()

        self.grad = 1.0
        for v in reversed(topo_list):
            v._backward()

    def __neg__(self):
        return self * -1

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * other**-1

    def __rtruediv__(self, other):
        return other * self**-1

    def __repr__(self):
        return f"Number(data={self.data}, grad={self.grad})"
