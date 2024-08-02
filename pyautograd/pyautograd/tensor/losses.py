from .tensor import Tensor
from .activations import ReLU
import numpy as np


class MSE:
    @staticmethod
    def apply(preds: Tensor, y: Tensor) -> Tensor:
        losses = preds - y
        batch_losses = (losses**2).sum(axis=-1) / losses.data[-1].size
        return batch_losses.sum() / batch_losses.data.size

    def __repr__():
        return "MSE Loss"


class HingeLoss:
    @staticmethod
    def apply(preds: Tensor, y: Tensor) -> Tensor:
        batch_losses = (1 + -y * preds).activation(ReLU).sum()
        return batch_losses.sum() / preds.data.size

    def __repr__():
        return "Hinge Loss"


class CrossEntropyLoss:
    @staticmethod
    def apply(preds: Tensor, y: Tensor) -> Tensor:
        batch_losses = -(y * preds.log()).sum(axis=-1).sum()
        return batch_losses / batch_losses.data.size

    def __repr__(self):
        return "CrossEntropyLoss"
