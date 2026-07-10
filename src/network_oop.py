"""
network_oop.py
~~~~~~~~~~~~~~

An object-oriented refactor of the original network.py.

Instead of one Network class that mixes architecture, math, and
training logic together, responsibilities are split across four
classes:

    Layer         - a single fully-connected layer: owns its weights
                    and biases, knows how to feed forward and how to
                    compute its own gradients.
    NeuralNetwork - a stack of Layers. Knows the *architecture* and
                    how to run feedforward/backprop across layers, but
                    knows nothing about how gradients get applied.
    Optimizer     - owns the update rule (here: vanilla SGD). Given
                    gradients, it decides how to change the weights.
                    Swapping in momentum/Adam later means writing a
                    new Optimizer subclass and nothing else changes.
    Trainer       - owns the training loop: epochs, mini-batching,
                    shuffling, calling the optimizer, and reporting
                    progress against test data.

This separation is the classic "single responsibility principle" -
each class has exactly one reason to change.
"""

import random
import numpy as np


#### Activation functions (pure, stateless -> free functions, not methods)
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    s = sigmoid(z)
    return s * (1 - s)


class Layer:
    """A single fully-connected layer.

    Owns its own weights and biases, and knows how to:
      - feed an input forward through itself
      - compute its own gradients given the error signal (delta)
        for its output, and propagate the error backward to whatever
        feeds it.
    """

    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        # Same init scheme as the original: Gaussian(0, 1)
        self.biases = np.random.randn(output_size, 1)
        self.weights = np.random.randn(output_size, input_size)

        # Populated during feedforward(); needed again during
        # compute_gradients(). This cache is what lets backprop avoid
        # recomputing the forward pass.
        self.last_input = None
        self.last_z = None
        self.last_activation = None

    def feedforward(self, a):
        """Compute this layer's output given input activation `a`."""
        self.last_input = a
        self.last_z = np.dot(self.weights, a) + self.biases
        self.last_activation = sigmoid(self.last_z)
        return self.last_activation

    def activation_prime(self):
        """sigmoid'(z) for the last z this layer computed."""
        return sigmoid_prime(self.last_z)

    def compute_gradients(self, delta):
        """Given delta = dC/dz for THIS layer's output, return:
            (nabla_b, nabla_w, propagated_error)
        where propagated_error = W^T . delta is the raw error to hand
        to the previous layer (that layer still needs to multiply by
        its own activation_prime() before using it as its delta).
        """
        if self.last_input is None:
            raise ValueError("feedforward() must be called before compute_gradients()")
        nabla_b = delta
        nabla_w = np.dot(delta, self.last_input.transpose())
        propagated_error = np.dot(self.weights.transpose(), delta)
        return nabla_b, nabla_w, propagated_error

    def apply_gradients(self, nabla_b, nabla_w, learning_rate, batch_size):
        """Directly nudge this layer's parameters. Normally called by
        an Optimizer rather than invoked by hand."""
        self.weights -= (learning_rate / batch_size) * nabla_w
        self.biases -= (learning_rate / batch_size) * nabla_b


class NeuralNetwork:
    """A feedforward network built from a stack of Layers.

    Knows the architecture (sizes) and how to run data through it
    forward and backward. Does NOT know how gradients get applied to
    weights - that's the Optimizer's job.
    """

    def __init__(self, sizes):
        """`sizes` e.g. [784, 30, 10] -> input layer of 784, one
        hidden layer of 30, output layer of 10."""
        self.sizes = sizes
        self.num_layers = len(sizes)
        self.layers = [
            Layer(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1)
        ]

    def feedforward(self, a):
        """Run input `a` through every layer in sequence."""
        for layer in self.layers:
            a = layer.feedforward(a)
        return a

    def backprop(self, x, y):
        """Return (nabla_b, nabla_w): layer-by-layer gradient lists
        for a single training example (x, y)."""
        # Forward pass, layer by layer (each layer caches what it needs)
        activation = self.feedforward(x)

        nabla_b = [None] * len(self.layers)
        nabla_w = [None] * len(self.layers)

        # Output layer error
        delta = self.cost_derivative(activation, y) * self.layers[-1].activation_prime()
        nb, nw, propagated = self.layers[-1].compute_gradients(delta)
        nabla_b[-1] = nb
        nabla_w[-1] = nw

        # Walk backward through the remaining layers
        for l in range(2, self.num_layers):
            layer = self.layers[-l]
            delta = propagated * layer.activation_prime()
            nb, nw, propagated = layer.compute_gradients(delta)
            nabla_b[-l] = nb
            nabla_w[-l] = nw

        return nabla_b, nabla_w

    def cost_derivative(self, output_activations, y):
        """dC/da for quadratic cost."""
        return output_activations - y

    def evaluate(self, test_data):
        """Count how many test inputs the network classifies correctly."""
        test_results = [
            (np.argmax(self.feedforward(x)), y) for (x, y) in test_data
        ]
        return sum(int(x == y) for (x, y) in test_results)


class Optimizer:
    """Owns the parameter-update rule. This is vanilla SGD; to add
    momentum, Adam, etc., write a new Optimizer subclass that
    overrides `update` - nothing in NeuralNetwork or Trainer changes."""

    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def update(self, network, nabla_b, nabla_w, batch_size):
        for layer, nb, nw in zip(network.layers, nabla_b, nabla_w):
            layer.apply_gradients(nb, nw, self.learning_rate, batch_size)


class Trainer:
    """Owns the training loop: epochs, shuffling, mini-batching, and
    progress reporting. Doesn't know anything about layer math - it
    just asks the network for gradients and hands them to the
    optimizer."""

    def __init__(self, network, optimizer):
        self.network = network
        self.optimizer = optimizer

    def train(self, training_data, epochs, mini_batch_size, test_data=None):
        training_data = list(training_data)
        n = len(training_data)

        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)

        for epoch in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k + mini_batch_size]
                for k in range(0, n, mini_batch_size)
            ]
            for mini_batch in mini_batches:
                self._train_on_mini_batch(mini_batch)

            if test_data:
                print(f"Epoch {epoch}: {self.network.evaluate(test_data)} / {n_test}")
            else:
                print(f"Epoch {epoch} complete")

    def _train_on_mini_batch(self, mini_batch):
        nabla_b = [np.zeros(layer.biases.shape) for layer in self.network.layers]
        nabla_w = [np.zeros(layer.weights.shape) for layer in self.network.layers]

        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.network.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        self.optimizer.update(self.network, nabla_b, nabla_w, len(mini_batch))