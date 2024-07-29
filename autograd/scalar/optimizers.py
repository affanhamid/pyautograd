import numpy as np
from .number import Number


class Optimizer:
    def __init__(self, model, loss_function):
        self.model = model
        self.loss_function = loss_function

    def get_batch(self, X, y, batch_size):
        if batch_size is None:
            return X, y
        else:
            ri = np.random.permutation(X.shape[0])[:batch_size]
            return X[ri], y[ri]

    def get_data_loss(self, preds, y):
        return self.loss_function.apply(preds, y)

    def run_model(self, Xb):
        return list(map(self.model, [list(map(Number, xrow)) for xrow in Xb]))

    def get_regularization_loss(self):

        alpha = 1e-4
        return alpha * sum((p * p for p in self.model.parameters()))

    def get_accuracy(self, preds, y):
        return [(yi > 0) == (p.data > 0) for yi, p in zip(y, preds)]

    def train(self, X, y, epochs, batch_size=None):
        for epoch in range(epochs):
            Xb, yb = self.get_batch(X, y, batch_size)
            preds = self.run_model(Xb)
            results = self.get_accuracy(preds, yb)
            accuracy = sum(results) / len(results)

            total_loss = self.get_data_loss(preds, yb) + self.get_regularization_loss()

            self.model.reset_gradients()
            total_loss.backward()

            learning_rate = 1.0 - 0.9 * epoch / 100
            self.model.update_parameters(learning_rate)

            if epoch % (epochs / 10) == 0:
                print(f"epoch {epoch} loss {total_loss.data}, accuracy {accuracy*100}%")

    def __repr__(self):
        return f"Stochastic Gradient Descent on \n\n {self.model} \n\n loss = {self.loss_function.__repr__()}"
