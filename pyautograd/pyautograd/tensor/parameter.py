import numpy as np
from pyautograd.tensor import Tensor


class Parameter(Tensor):
    def __init__(self, *shape) -> None:
        super().__init__(data=np.random.randn(*shape), requires_grad=True)
