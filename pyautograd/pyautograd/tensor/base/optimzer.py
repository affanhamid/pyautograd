import numpy as np
from .module import Module


class Optimizer:
    def __init__(self, model: Module, loss_function):
        self.model = model
        self.loss_function = loss_function

    def update_parameters(self, module: Module, alpha: float) -> None:
        for parameter in module.parameters():
            parameter -= parameter.grad * alpha
