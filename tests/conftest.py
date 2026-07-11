"""
Shared pytest fixtures.

All fixtures here use a small synthetic network/dataset rather than
the real MNIST architecture and data. That's deliberate: unit tests
should run in a fraction of a second and not depend on mnist.pkl.gz
existing on disk. The real network gets exercised by the actual
training script and the live API, not by these tests.
"""

import numpy as np
import pytest

from digit_classifier.model.network import NeuralNetwork


@pytest.fixture
def small_network():
    """A tiny network - small enough that exhaustive gradient checking
    and fast training are both cheap, but with a real hidden layer so
    it exercises the same code paths (multiple layers, backprop's
    layer-to-layer error propagation) as the real 784-30-10 network."""
    np.random.seed(0)
    return NeuralNetwork([3, 4, 2])


@pytest.fixture
def single_example():
    """One (x, y) pair, shaped like a real MNIST example (a column
    vector input, a column vector target) but tiny."""
    np.random.seed(1)
    x = np.random.randn(3, 1)
    y = np.random.rand(2, 1)
    return x, y


@pytest.fixture
def tiny_dataset():
    """A handful of fixed examples with a genuinely learnable rule
    (which half-space x falls in), so a convergence test is actually
    testing something - not just checking that loss can memorize
    noise."""
    rng = np.random.RandomState(42)
    examples = []
    for _ in range(16):
        x = rng.randn(3, 1)
        label = 0 if x.sum() > 0 else 1
        y = np.zeros((2, 1))
        y[label] = 1.0
        examples.append((x, y))
    return examples
