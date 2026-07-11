"""
Structural regression tests: do feedforward() and backprop() produce
arrays of the right shape?

These are deliberately separate from the gradient-check test. Shape
bugs (a bad reshape, a transpose in the wrong place, wiring the wrong
layer's cache into a computation) and VALUE bugs (a sign error, wrong
scalar) are different failure modes. A shape test catches the former
in milliseconds without doing any perturb-and-measure work; it won't
catch the latter, which is what test_gradients.py is for.
"""

import numpy as np
import pytest

from digit_classifier.model.network import NeuralNetwork


@pytest.mark.parametrize("sizes", [
    [3, 4, 2],       # one hidden layer, matches the small_network fixture's shape
    [5, 6, 3, 2],    # two hidden layers, checks backprop's loop over >1 hidden layer
])
def test_feedforward_output_shape(sizes):
    np.random.seed(0)
    net = NeuralNetwork(sizes)
    x = np.random.randn(sizes[0], 1)

    output = net.feedforward(x)

    assert output.shape == (sizes[-1], 1)


@pytest.mark.parametrize("sizes", [
    [3, 4, 2],
    [5, 6, 3, 2],
])
def test_backprop_gradient_shapes_match_parameters(sizes):
    np.random.seed(0)
    net = NeuralNetwork(sizes)
    x = np.random.randn(sizes[0], 1)
    y = np.random.rand(sizes[-1], 1)

    nabla_b, nabla_w = net.backprop(x, y)

    assert len(nabla_b) == len(net.layers)
    assert len(nabla_w) == len(net.layers)

    for layer, nb, nw in zip(net.layers, nabla_b, nabla_w):
        assert nb.shape == layer.biases.shape
        assert nw.shape == layer.weights.shape
