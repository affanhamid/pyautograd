from .tensor import Tensor, Dependency
import numpy as np
from typing import List


def _tensor_sum(t: Tensor, axis) -> Tensor:
    """Returns the sum of all elements of the tensor"""

    data = t.data.sum(axis=axis)
    requires_grad = t.requires_grad

    if requires_grad:

        def grad_fn(grad: np.ndarray) -> np.ndarray:
            """
            Suppose,
                [[1, 2, 3],
            T =  [4, 5, 6],
                 [,7, 8, 9]]
            and
            T2 = T.sum() = t11 + t12 ... + t33

            Now suppose that the gradient of the loss w.r.t T2 is dT2

            dt11 = dT2, dt12 = dT2, ... dt33 = dT2
            So,
            dtij = 1 * dT2

            or,

                [[dT2, dT2, dT2],
            dT = [dT2, dT2, dT2],
                 [dT2, dT2, dT2]]
            """
            return grad * np.ones_like(t.data)

        depends_on = [Dependency(t, grad_fn)]
    else:
        depends_on = []

    return Tensor(data=data, requires_grad=requires_grad, depends_on=depends_on)


def _add(t1: Tensor, t2: Tensor) -> Tensor:
    data = t1.data + t2.data
    requires_grad = t1.requires_grad or t2.requires_grad
    depends_on: List[Dependency] = []

    if t1.requires_grad:

        def t1_grad_fn(grad: np.ndarray) -> np.ndarray:
            ndims_added = grad.data.ndim - t1.data.ndim
            for _ in range(ndims_added):
                grad = grad.sum(axis=0)

            for i, dim in enumerate(t1.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)
            return grad

        depends_on.append(Dependency(t1, t1_grad_fn))

    if t2.requires_grad:

        def t2_grad_fn(grad: np.ndarray) -> np.ndarray:
            ndims_added = grad.data.ndim - t2.data.ndim
            for _ in range(ndims_added):
                grad = grad.sum(axis=0)

            for i, dim in enumerate(t2.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)
            return grad

        depends_on.append(Dependency(t2, t2_grad_fn))

    return Tensor(data=data, requires_grad=requires_grad, depends_on=depends_on)


def _mul(t1: Tensor, t2: Tensor) -> Tensor:
    data = t1.data * t2.data
    requires_grad = t1.requires_grad or t2.requires_grad

    depends_on: List[Dependency] = []
    if t1.requires_grad:

        def t1_grad_fn(grad: np.ndarray) -> np.ndarray:
            grad = grad * t2.data
            ndims_added = grad.data.ndim - t1.data.ndim
            for _ in range(ndims_added):
                grad = grad.sum(axis=0)

            for i, dim in enumerate(t1.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)
            return grad

        depends_on.append(Dependency(t1, t1_grad_fn))

    if t2.requires_grad:

        def t2_grad_fn(grad: np.ndarray) -> np.ndarray:
            grad = grad * t1.data
            ndims_added = grad.data.ndim - t2.data.ndim
            for _ in range(ndims_added):
                grad = grad.sum(axis=0)

            for i, dim in enumerate(t2.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)
            return grad

        depends_on.append(Dependency(t2, t2_grad_fn))

    return Tensor(data=data, requires_grad=requires_grad, depends_on=depends_on)


def _neg(t: Tensor) -> Tensor:
    data = -t.data
    requires_grad = t.requires_grad

    if requires_grad:
        depends_on = [Dependency(t, lambda x: -x)]
    else:
        depends_on = []

    return Tensor(data, requires_grad=requires_grad, depends_on=depends_on)


def _matmul(t1: Tensor, t2: Tensor) -> Tensor:
    data = t1.data @ t2.data
    requires_grad = t1.requires_grad or t2.requires_grad

    depends_on: List[Dependency] = []
    if t1.requires_grad:

        def t1_grad_fn(grad: np.ndarray) -> np.ndarray:
            return grad @ t2.data.T

        depends_on.append(Dependency(t1, t1_grad_fn))

    if t2.requires_grad:

        def t2_grad_fn(grad: np.ndarray) -> np.ndarray:
            return t1.data.T @ grad

        depends_on.append(Dependency(t2, t2_grad_fn))

    return Tensor(data=data, requires_grad=requires_grad, depends_on=depends_on)


def _slice(t: Tensor, *idxs) -> Tensor:
    data = t.data[*idxs]
    requires_grad = t.requires_grad

    if requires_grad:

        def grad_fn(grad: np.ndarray) -> np.ndarray:
            bigger_grad = np.zeros_like(t.data)
            bigger_grad[*idxs] = grad
            return bigger_grad

        depends_on = Dependency(t, grad_fn)
    else:
        depends_on = []

    return Tensor(data, requires_grad, depends_on)
