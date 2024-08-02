import inspect
from typing import Iterator
from pyautograd.tensor import Tensor


class Module:
    def parameters(self) -> Iterator[Tensor]:
        for name, value in inspect.getmembers(self):
            if isinstance(value, Tensor):
                yield value
            elif isinstance(value, Module):
                yield from value.parameters()

    def zero_grad(self) -> None:
        for parameter in self.parameters():
            parameter.zero_grad()
