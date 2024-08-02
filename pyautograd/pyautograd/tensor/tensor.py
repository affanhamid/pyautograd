import numpy as np
from typing import List, NamedTuple, Callable, Optional, Union


class Dependency(NamedTuple):
    tensor: "Tensor"
    grad_fn: Callable[[np.ndarray], np.ndarray]


Arrayable = Union[float, list, np.ndarray]


def ensure_array(arrayable: Arrayable) -> np.ndarray:
    if isinstance(arrayable, np.ndarray):
        return arrayable
    else:
        return np.array(arrayable)


Tensorable = Union["Tensor", float, np.ndarray]


def ensure_tensor(tensorable: Tensorable) -> "Tensor":
    if isinstance(tensorable, Tensor):
        return tensorable
    else:
        return Tensor(tensorable)


class Tensor:
    def __init__(
        self,
        data: Arrayable,
        requires_grad: bool = False,
        depends_on: List[Dependency] = [],
    ) -> None:
        self._data = ensure_array(data)
        self.requires_grad = requires_grad
        self.depends_on = depends_on

        self.shape = self._data.shape
        self.grad: Optional["Tensor"] = None

        if self.requires_grad:
            self.zero_grad()

    @property
    def data(self) -> np.ndarray:
        return self._data

    @data.setter
    def data(self, new_data: np.ndarray) -> None:
        self._data = new_data
        self.grad = None

    def zero_grad(self) -> None:
        self.grad = Tensor(data=np.zeros_like(self.data))

    def __repr__(self) -> str:
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"

    def sum(self, axis=0) -> "Tensor":
        from .operations import Sum

        return Sum.forward(self, axis)

    def __add__(self, other) -> "Tensor":
        from .operations import Add

        return Add.forward(self, ensure_tensor(other))

    def __radd__(self, other) -> "Tensor":
        return self + other

    def __iadd__(self, other) -> "Tensor":
        self.data = self.data + ensure_tensor(other).data
        return self

    def __isub__(self, other) -> "Tensor":
        self.data = self.data - ensure_tensor(other).data
        return self

    def __imul__(self, other) -> "Tensor":
        self.data = self.data * ensure_tensor(other).data
        return self

    def __mul__(self, other) -> "Tensor":
        from .operations import ElementWiseMultiply

        return ElementWiseMultiply.forward(self, ensure_tensor(other))

    def __rmul__(self, other) -> "Tensor":
        return self * other

    def __matmul__(self, other) -> "Tensor":
        from .operations import MatrixMultiplication

        return MatrixMultiplication.forward(self, other)

    def __neg__(self) -> "Tensor":
        from .operations import Negative

        return Negative.forward(self)

    def __sub__(self, other) -> "Tensor":
        from .operations import Subtract

        return Subtract.forward(self, ensure_tensor(other))

    def __rsub__(self, other) -> "Tensor":
        return -(self - other)

    def __getitem__(self, *idxs) -> "Tensor":
        from .operations import Slice

        return Slice.forward(self, *idxs)

    def __pow__(self, other) -> "Tensor":
        from .operations import Power

        return Power.forward(self, other)

    def __truediv__(self, other) -> "Tensor":
        return self * (1 / other)

    def activation(self, activationClass) -> "Tensor":
        return activationClass.forward(self)

    def log(self) -> "Tensor":
        from .operations import Log

        return Log.forward(self)

    def exp(self) -> "Tensor":
        from .operations import Exp

        return Exp.forward(self)

    def backward(self, grad: "Tensor" = None) -> None:
        assert (
            self.requires_grad
        ), "Called backwards on a tensor that doesn't require grad"

        if grad is None:
            grad = Tensor(np.ones_like(self.data))

        self.grad.data = self.grad.data + grad.data
        for dependency in self.depends_on:
            backward_grad = dependency.grad_fn(grad.data)
            dependency.tensor.backward(Tensor(backward_grad))
