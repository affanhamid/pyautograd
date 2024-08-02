from .base.module import Module
import numpy as np
from .tensor import Tensor


class Layer(Module):

    def __init__(self, input_size, layer_size, activation=None):
        self.w = Tensor(np.random.randn(input_size, layer_size), requires_grad=True)
        self.b = Tensor(np.random.randn(layer_size), requires_grad=True)
        self.activation = activation

    def forward(self, x: Tensor):
        z = x @ self.w + self.b

        return self.activation.forward(z) if self.activation else z


class Network(Module):

    def __init__(self, layers):
        self.layers = layers

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
