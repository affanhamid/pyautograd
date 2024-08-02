from .tensor import Tensor, Dependency
import numpy as np


class Sum:
    @staticmethod
    def forward(t: Tensor, axis=None) -> Tensor:
        """Returns the sum of elements of the tensor using numpy's sum"""
        """
            Suppose,
                 [[1, 2, 3],
            T =   [4, 5, 6],
                  [,7, 8, 9]]
            
            Then, its forward prop will be:
            T2 = T.sum() = t11 + t12 ... + t33
        """
        data = t.data.sum(axis=axis)
        requires_grad = t.requires_grad

        depends_on = (
            [Dependency(t, lambda grad: Sum.backward(grad, t))] if requires_grad else []
        )

        return Tensor(data=data, requires_grad=requires_grad, depends_on=depends_on)

    @staticmethod
    def backward(grad: np.ndarray, t: Tensor) -> np.ndarray:
        """
        Suppose that the gradient of the loss w.r.t T2 is dT2

        dt11 = dT2, dt12 = dT2, ... dt33 = dT2
        So,
        dtij = 1 * dT2

        or,

             [[dT2, dT2, dT2],
        dT =  [dT2, dT2, dT2],
              [dT2, dT2, dT2]]
        """
        return grad * np.ones_like(t.data)


class Add:
    @staticmethod
    def forward(t1: Tensor, t2: Tensor) -> Tensor:
        """
        Suppose,
             [[1, 2, 3],
        T =   [4, 5, 6],
              [7, 8, 9]]

        Then, its forward prop will be:
        T2 = T.sum() = t11 + t12 ... + t33
        """
        data = t1.data + t2.data
        requires_grad = t1.requires_grad or t2.requires_grad
        depends_on = []

        if t1.requires_grad:
            depends_on.append(Dependency(t1, lambda grad: Add.backward(grad, t1)))
        if t2.requires_grad:
            depends_on.append(Dependency(t2, lambda grad: Add.backward(grad, t2)))

        return Tensor(data=data, requires_grad=requires_grad, depends_on=depends_on)

    @staticmethod
    def backward(grad: np.ndarray, t: Tensor) -> None:
        """
        Suppose C = A + B
        if dC is the gradient of the loss w.r.t C
        Then dA = dC and dB = dC

        But, numpy allows for broadcasting(https://numpy.org/doc/stable/user/basics.broadcasting.html)
        So we also adjust for the added dimensions due to the two types of broadcasting
        """
        ndims_added = grad.data.ndim - t.data.ndim
        for _ in range(ndims_added):
            grad = grad.sum(axis=0)

        for i, dim in enumerate(t.shape):
            if dim == 1:
                grad = grad.sum(axis=i, keepdims=True)
        return grad


class ElementWiseMultiply:
    @staticmethod
    def forward(t1: Tensor, t2: Tensor) -> Tensor:
        data = t1.data * t2.data
        requires_grad = t1.requires_grad or t2.requires_grad
        depends_on = []

        if t1.requires_grad:
            depends_on.append(
                Dependency(t1, lambda grad: ElementWiseMultiply.backward(grad, t1, t2))
            )
        if t2.requires_grad:
            depends_on.append(
                Dependency(t2, lambda grad: ElementWiseMultiply.backward(grad, t2, t1))
            )

        return Tensor(data=data, requires_grad=requires_grad, depends_on=depends_on)

    @staticmethod
    def backward(grad: np.ndarray, t1: Tensor, t2: Tensor) -> None:
        grad = grad * t2.data
        ndims_added = grad.data.ndim - t1.data.ndim
        for _ in range(ndims_added):
            grad = grad.sum(axis=0)

        for i, dim in enumerate(t1.shape):
            if dim == 1:
                grad = grad.sum(axis=i, keepdims=True)
        return grad


class Negative:
    @staticmethod
    def forward(t: Tensor) -> Tensor:
        data = -t.data
        requires_grad = t.requires_grad

        if requires_grad:
            depends_on = [Dependency(t, lambda grad: -grad)]
        else:
            depends_on = []

        return Tensor(data, requires_grad=requires_grad, depends_on=depends_on)


class Subtract:
    @staticmethod
    def forward(t1: Tensor, t2: Tensor) -> Tensor:
        data = t1.data - t2.data
        requires_grad = t1.requires_grad or t2.requires_grad
        depends_on = []

        if t1.requires_grad:
            depends_on.append(Dependency(t1, lambda grad: Add.backward(grad, t1)))
        if t2.requires_grad:
            depends_on.append(Dependency(t2, lambda grad: -Add.backward(grad, t2)))

        return Tensor(data=data, requires_grad=requires_grad, depends_on=depends_on)


class MatrixMultiplication:
    @staticmethod
    def forward(t1: Tensor, t2: Tensor) -> Tensor:
        data = t1.data @ t2.data
        requires_grad = t1.requires_grad or t2.requires_grad
        depends_on = []

        if t1.requires_grad:
            depends_on.append(Dependency(t1, lambda grad: grad @ t2.data.T))
        if t2.requires_grad:
            depends_on.append(Dependency(t2, lambda grad: t1.data.T @ grad))

        return Tensor(data=data, requires_grad=requires_grad, depends_on=depends_on)


class Slice:
    @staticmethod
    def forward(t: Tensor, *idxs) -> Tensor:
        data = t.data[*idxs]
        requires_grad = t.requires_grad

        depends_on = (
            [Dependency(t, lambda grad: Slice.backward(grad, t, *idxs))]
            if requires_grad
            else []
        )

        return Tensor(data, requires_grad, depends_on)

    @staticmethod
    def backward(grad: np.ndarray, t: Tensor, *idxs) -> np.ndarray:
        bigger_grad = np.zeros_like(t.data)
        bigger_grad[*idxs] = grad
        return bigger_grad


class Log:
    @staticmethod
    def forward(t: Tensor) -> Tensor:
        data = np.log(t.data)
        requires_grad = t.requires_grad

        depends_on = (
            [Dependency(t, lambda grad: Log.backward(grad, t))] if requires_grad else []
        )

        return Tensor(data, requires_grad, depends_on)

    @staticmethod
    def backward(grad: np.ndarray, t: Tensor) -> np.ndarray:
        return grad / t.data


class Exp:
    @staticmethod
    def forward(t: Tensor) -> Tensor:
        data = np.exp(t.data)
        requires_grad = t.requires_grad

        depends_on = (
            [Dependency(t, lambda grad: Log.backward(grad, t))] if requires_grad else []
        )

        return Tensor(data, requires_grad, depends_on)

    @staticmethod
    def backward(grad: np.ndarray, t: Tensor) -> np.ndarray:
        return grad * t.data


class Power:
    @staticmethod
    def forward(t: Tensor, value: float) -> Tensor:
        data = np.power(t.data, value)
        requires_grad = t.requires_grad

        depends_on = (
            [Dependency(t, lambda grad: Power.backward(grad, t, value))]
            if requires_grad
            else []
        )

        return Tensor(data, requires_grad, depends_on)

    @staticmethod
    def backward(grad: np.ndarray, t: Tensor, value: float) -> np.ndarray:
        return grad * value * np.power(t.data, value - 1)
