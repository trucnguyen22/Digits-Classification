"""
gradient_checker.py
~~~~~~~~~~~~~~~~~~~

Numerical gradient checking: an independent way to verify that
NeuralNetwork.backprop() computes correct gradients, WITHOUT trusting
backprop's own math. It works from the raw definition of a derivative
instead of the chain rule, so a bug in backprop can't hide from it.

For a single parameter (one weight or one bias), we estimate its
gradient using the centered difference formula:

    numeric_grad = (cost(param + eps) - cost(param - eps)) / (2 * eps)

and compare it against the analytic gradient backprop() produced for
that same parameter, using relative error so parameters of very
different scales are compared fairly:

    relative_error = |analytic - numeric| / (|analytic| + |numeric|)

Checking every parameter is O(num_params) feedforward passes, which
gets slow on larger networks. GradientChecker supports checking a
random sample instead of the full parameter set - see `sample_size`.
"""

import random

import numpy as np


def quadratic_cost(output_activations, y):
    """C = 1/2 * sum((a - y)^2). Matches NeuralNetwork.cost_derivative,
    which is dC/da = a - y for this exact cost function - the two must
    stay consistent, since we're checking backprop's gradient of THIS
    cost, not some other one."""
    return 0.5 * float(np.sum((output_activations - y) ** 2))


class GradientChecker:
    """Compares analytic gradients (from NeuralNetwork.backprop) against
    numerically-estimated gradients (from perturb-and-measure), for a
    single (x, y) example."""

    def __init__(self, epsilon=1e-4, relative_error_threshold=1e-7):
        # epsilon: the perturbation size. Too large -> the finite
        # difference stops approximating the true local slope. Too
        # small -> float64 rounding error dominates when subtracting
        # two nearly-equal costs. 1e-4 is the standard sweet spot.
        self.epsilon = epsilon
        self.relative_error_threshold = relative_error_threshold

    def _all_parameter_refs(self, network):
        """Every (layer_index, kind, row, col) address of a learnable
        parameter in the network, where kind is 'weights' or 'biases'."""
        refs = []
        for l, layer in enumerate(network.layers):
            for row in range(layer.weights.shape[0]):
                for col in range(layer.weights.shape[1]):
                    refs.append((l, "weights", row, col))
            for row in range(layer.biases.shape[0]):
                refs.append((l, "biases", row, 0))
        return refs

    def _cost_with_perturbation(self, network, x, y, layer_idx, kind, row, col, delta):
        """Nudge ONE parameter by `delta`, run a plain feedforward (no
        backprop - we don't want backprop's own math anywhere near this
        measurement), read off the cost, then restore the parameter."""
        layer = network.layers[layer_idx]
        param_array = getattr(layer, kind)

        original_value = param_array[row, col]
        param_array[row, col] = original_value + delta
        activation = network.feedforward(x)
        cost = quadratic_cost(activation, y)
        param_array[row, col] = original_value  # always restore, even if we raise later

        return cost

    def check(self, network, x, y, sample_size=None, seed=None):
        """Run the check for one (x, y) example.

        sample_size=None checks every parameter. Otherwise, checks a
        random sample of that many parameters (without replacement) -
        useful for larger networks where checking everything would be
        too slow to run routinely.

        Returns a list of dicts, one per checked parameter:
            {layer, kind, index, analytic, numeric, relative_error}
        """
        analytic_nabla_b, analytic_nabla_w = network.backprop(x, y)

        all_refs = self._all_parameter_refs(network)
        if sample_size is not None and sample_size < len(all_refs):
            rng = random.Random(seed)
            refs_to_check = rng.sample(all_refs, sample_size)
        else:
            refs_to_check = all_refs

        results = []
        for (layer_idx, kind, row, col) in refs_to_check:
            cost_plus = self._cost_with_perturbation(
                network, x, y, layer_idx, kind, row, col, self.epsilon
            )
            cost_minus = self._cost_with_perturbation(
                network, x, y, layer_idx, kind, row, col, -self.epsilon
            )
            numeric_grad = (cost_plus - cost_minus) / (2 * self.epsilon)

            if kind == "weights":
                analytic_grad = analytic_nabla_w[layer_idx][row, col]
            else:
                analytic_grad = analytic_nabla_b[layer_idx][row, col]

            denom = abs(analytic_grad) + abs(numeric_grad)
            relative_error = 0.0 if denom == 0 else abs(analytic_grad - numeric_grad) / denom

            results.append({
                "layer": layer_idx,
                "kind": kind,
                "index": (row, col),
                "analytic": float(analytic_grad),
                "numeric": float(numeric_grad),
                "relative_error": float(relative_error),
            })

        return results

    def max_relative_error(self, results):
        return max(r["relative_error"] for r in results) if results else 0.0

    def passed(self, results):
        return self.max_relative_error(results) < self.relative_error_threshold
