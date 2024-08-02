import numpy as np
from .base.optimzer import Optimizer
from .base.module import Module


class SGD(Optimizer):
    def __init__(self, model: Module):
        self.model = model

    def get_batch(self, X, y, batch_size):
        if batch_size is None:
            return X, y
        else:
            ri = np.random.permutation(X.shape[0])[:batch_size]
            return X[ri], y[ri]

    def step(self, X, y, learning_rate, loss_function, batch_size=None):
        Xb, yb = self.get_batch(X, y, batch_size)
        preds = self.model.forward(Xb)

        self.model.zero_grad()
        loss = loss_function.apply(preds, yb)
        loss.backward()

        self.update_parameters(self.model, learning_rate)
        return loss

    def __repr__(self):
        return f"Stochastic Gradient Descent on \n\n {self.model} \n\n loss = {self.loss_function.__repr__()}"
