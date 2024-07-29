# Autograd

Autograd is an auto-differentiation library for scalars, designed to facilitate the implementation of machine learning models. It provides a simple interface for defining and training neural networks using automatic differentiation.

## Features

- **Automatic Differentiation**: Compute gradients automatically for scalar operations.
- **Neural Network Components**: Build neural networks using layers and neurons.
- **Optimizers**: Implement optimization algorithms (Currently only Stochastic Gradient Descent)
- **Loss Functions**: Support for common loss functions (Currently only Hinge Loss)

## Installation

To install the library, clone the repository:

```bash
pip install py-autograd
```

The only requirement is `numpy`

## Usage

### Importing the Library

You can import the necessary components from the library as follows:

```python
from autograd.scalar import Number, Layer, Network
from autograd.scalar.activations import ReLU
from autograd.scalar.losses import HingeLoss
from autograd.scalar.optimizers import Optimizer
```

### Using the Number class

The Number class wraps around python's integers and floats. It has all the building blocks including addition, multiplication, exponentiation, power, etc. During each calculation, the Number class stores the function to get the local gradient at the point.

```python
a = Number(3.0)
b = 10

c = a**10
d = c.exp()
e = d / a
f = 2*e
```

The number class also includes activation functions:

```python
from autograd.scalar.activations import ReLU
a = Number(-5.0)
b = a.activation(ReLU)
```

Finally, running the `.backward()` function on any `Number` object will calculate the gradients of all the variables before it. These gradients can be accessed by the `.grad` attribute

```python
f.backward()
a.grad
```


### Defining a Neural Network

You can also define a multi-layered perceptron (MLP) using the `Network` and `Layer` classes:

```python
model = Network([
    Layer(2, 16, ReLU()),
    Layer(16, 16, ReLU()),
    Layer(16, 1)
])
```

### Training the Model

To train the model, create an instance of the `Optimizer` class and call the `train` method. Currently, the library uses Stochastic Gradient Descent to train the model

```python
optimizer = Optimizer(model, HingeLoss)
optimizer.train(X, y, epochs=100, batch_size=32)
```


## Example

An example of using the library can be found in the `demo.ipynb` Jupyter notebook. This notebook demonstrates how to create a dataset, define a model, train it, and visualize the results.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.