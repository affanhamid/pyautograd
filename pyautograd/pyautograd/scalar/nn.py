import random
from .number import Number


class Neuron:

    def __init__(self, input_size, activation=None):
        self.w = [Number(random.uniform(-1, 1)) for _ in range(input_size)]
        self.b = Number(0)
        self.activation = activation

    def __call__(self, input_vec):
        z = sum((wi * xi for wi, xi in zip(self.w, input_vec)), self.b)
        return z.activation(self.activation) if self.activation else z

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"Neuron(input={len(self.w)}, {self.activation.__repr__() if self.activation else 'Linear'})"


class Layer:

    def __init__(self, input_size, layer_size, activation=None):
        self.neurons = [Neuron(input_size, activation) for _ in range(layer_size)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer({len(self.neurons[0].w)}, {len(self.neurons)}, {self.neurons[0].activation.__repr__() if self.neurons[0].activation else "Linear"})"


class Network:

    def __init__(self, layers):
        self.layers = layers

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def reset_gradients(self):
        for p in self.parameters():
            p.grad = 0

    def update_parameters(self, learning_rate):
        for p in self.parameters():
            p.data -= learning_rate * p.grad

    def __repr__(self):
        return f" Multi Layered Perceptron:\n\t{'\n\t'.join(str(layer) for layer in self.layers)}\n number of parameters = {len(self.parameters())}"
