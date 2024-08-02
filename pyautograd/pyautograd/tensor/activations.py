import numpy as np
from .tensor import Tensor
from .base.activation import Activation


class Tanh(Activation):
    @staticmethod
    def run_func(x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

    @staticmethod
    def backward(grad: np.ndarray, forward_data: np.ndarray) -> np.ndarray:
        return grad * (1 - forward_data * forward_data)


class ReLU(Activation):
    @staticmethod
    def run_func(x: Tensor) -> Tensor:
        return np.maximum(x.data, 0)

    @staticmethod
    def backward(grad: np.ndarray, forward_data: np.ndarray) -> Tensor:
        return grad * (forward_data > 0)


class Sigmoid(Activation):
    @staticmethod
    def run_func(x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def backward(grad: np.ndarray, forward_data: np.ndarray) -> np.ndarray:
        return grad * forward_data * (1 - forward_data)


class Softmax(Activation):
    @staticmethod
    def run_func(t: np.ndarray) -> np.ndarray:
        exp_t = np.exp(t - np.max(t, axis=-1, keepdims=True))  # for numerical stability
        return exp_t / np.sum(exp_t, axis=-1, keepdims=True)

    @staticmethod
    def backward(grad: np.ndarray, forward_data: np.ndarray) -> np.ndarray:
        grad_output = np.empty_like(grad)
        for i, (g, s) in enumerate(zip(grad, forward_data)):
            s = s.reshape(-1, 1)
            jacobian_m = np.diagflat(s) - np.dot(s, s.T)
            grad_output[i] = np.dot(jacobian_m, g)
        return grad_output
