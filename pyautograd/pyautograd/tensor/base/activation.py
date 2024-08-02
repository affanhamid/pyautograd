from abc import ABC, abstractmethod
import numpy as np
from pyautograd.tensor.tensor import Tensor, Dependency


class Activation(ABC):
    @classmethod
    def forward(cls, t: Tensor) -> Tensor:
        data = cls.run_func(t.data)
        requires_grad = t.requires_grad
        depends_on = (
            [Dependency(t, lambda grad: cls.backward(grad, data))]
            if requires_grad
            else []
        )
        return Tensor(data, requires_grad, depends_on)

    @staticmethod
    @abstractmethod
    def run_func(t: np.ndarray) -> np.ndarray:
        pass

    @staticmethod
    @abstractmethod
    def backward(grad: np.ndarray, t: np.ndarray) -> np.ndarray:
        pass
