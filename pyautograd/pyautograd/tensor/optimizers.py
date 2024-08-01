from pyautograd.module import Module


class SGD:
    def __init__(self, alpha: float = 0.01) -> None:
        self.alpha = alpha

    def step(self, module: Module) -> None:
        for parameter in module.parameters():
            parameter -= parameter.grad * self.alpha
