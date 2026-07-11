"""
Convergence sanity test: does the loss actually go down as the
network trains?

This is an integration-level check that the gradient-check and shape
tests can't provide on their own. Both of those verify a single
backprop() call in isolation - they wouldn't catch a bug in
Optimizer.update() (e.g. a flipped sign that pushes weights the wrong
direction), because that code runs *after* backprop and isn't part of
what either of those tests exercises. Actually training for a few
epochs and confirming loss trends down is what catches that class of
bug.
"""

import numpy as np

from digit_classifier.model.network import Optimizer, Trainer
from digit_classifier.diagnostics.gradient_checker import quadratic_cost


def _average_loss(network, dataset):
    return sum(quadratic_cost(network.feedforward(x), y) for x, y in dataset) / len(dataset)


def test_loss_decreases_over_epochs(small_network, tiny_dataset):
    optimizer = Optimizer(learning_rate=1.0)
    trainer = Trainer(small_network, optimizer)

    losses = [_average_loss(small_network, tiny_dataset)]
    for _ in range(30):
        trainer.train(tiny_dataset, epochs=1, mini_batch_size=4)
        losses.append(_average_loss(small_network, tiny_dataset))

    # SGD on mini-batches is noisy epoch-to-epoch, so we don't require
    # strictly monotonic decrease. Instead compare the average loss
    # over the first few epochs to the average over the last few -
    # the overall trend should clearly be downward.
    early_average = np.mean(losses[:3])
    late_average = np.mean(losses[-3:])

    assert late_average < early_average, (
        f"Loss did not improve: early epochs averaged {early_average:.4f}, "
        f"last epochs averaged {late_average:.4f}"
    )
