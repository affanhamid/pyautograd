from .activations import ReLU


class HingeLoss:
    @staticmethod
    def apply(preds, y):
        losses = [(1 + -yi * p).activation(ReLU) for yi, p in zip(y, preds)]
        return sum(losses) * (1.0 / len(losses))

    def __repr__():
        return "Hinge Loss"
