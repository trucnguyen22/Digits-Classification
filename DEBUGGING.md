# Debugging: Verifying Backpropagation Correctness

## Why this exists

`NeuralNetwork.backprop()` computes gradients analytically, by hand-coding
the chain rule layer by layer. That's a lot of places to introduce a subtle
bug (a transpose in the wrong spot, a sign error, the wrong layer's cached
`z` used at the wrong step) — and the dangerous part is that a wrong
gradient often still trains the network *somewhat*, just worse than it
should. Loss going down is not proof that backprop is correct. This
document describes an independent check that doesn't rely on backprop's
own math at all, and what it actually found when run against this codebase.

## Method: numerical gradient checking

For a single parameter (one weight or one bias), we can estimate its
gradient directly from the definition of a derivative — the local slope of
the cost function — without touching backprop:

```
numeric_grad = (cost(param + eps) - cost(param - eps)) / (2 * eps)
```

This is the *centered difference* formula. Centering it (perturbing both up
and down, rather than just one direction) cancels out more of the
approximation error than a one-sided version.

We then compare that against the analytic gradient `backprop()` produced
for the same parameter, using **relative error** rather than raw
difference, since different parameters can have gradients of very
different magnitudes:

```
relative_error = |analytic - numeric| / (|analytic| + |numeric|)
```

Implementation: `src/digit_classifier/diagnostics/gradient_checker.py`
(`GradientChecker` class).

## Design decisions

**epsilon = 1e-4.** Too large and the finite-difference approximation stops
tracking the true local slope (the cost function is curved, not linear).
Too small and float64 rounding error dominates, since we're subtracting two
nearly-identical cost values. `1e-4` is the standard middle ground for
double precision.

**Small synthetic network, not the real 784-30-10 architecture.** The
regression test (`tests/test_gradients.py`) runs against a tiny `[3, 4, 2]`
network with a single random example, not the real MNIST-sized network.
This keeps the check fast enough to run on every test invocation, removes
any dependency on `mnist.pkl.gz` being present, and still exercises the
same code paths — multiple layers, backprop's layer-to-layer error
propagation, the sigmoid activation derivative — as the real network.

**Sampling support for larger networks.** `GradientChecker.check()` accepts
an optional `sample_size` to check a random subset of parameters instead of
all of them. The `[3, 4, 2]` network only has 26 parameters, so the test
suite checks all of them exhaustively. Pointed at a much larger
architecture, exhaustively checking every parameter (two extra feedforward
passes each) would get slow fast; sampling makes the same tool usable at
that scale.

## Result

Running `GradientChecker` against the `[3, 4, 2]` network, checking all 26
parameters against a single random example:

```
num params checked: 26
max relative error: 1.20e-08
threshold: 1e-07
passed: True
```

**No discrepancy was found.** The analytic gradients from `backprop()`
match the independently-computed numerical gradients to within floating
point precision. This is the expected, correct outcome — it confirms the
current implementation of `Layer.compute_gradients()` and
`NeuralNetwork.backprop()` is mathematically correct, rather than "training
loss looks reasonable so it's probably fine."

If a future change to the backprop math introduces a real bug, this same
check (locked in as `tests/test_gradients.py`) will fail with a relative
error far above `1e-7`, and the `layer`/`kind`/`index` fields in each
result pinpoint exactly which parameter's gradient is wrong.

## Reproducing this

```
pip install -e ".[dev]"
pytest tests/test_gradients.py -v
```

Or run the check directly for more detail on individual parameters:

```python
from digit_classifier.model.network import NeuralNetwork
from digit_classifier.diagnostics.gradient_checker import GradientChecker
import numpy as np

net = NeuralNetwork([3, 4, 2])
x = np.random.randn(3, 1)
y = np.random.rand(2, 1)

checker = GradientChecker(epsilon=1e-4)
results = checker.check(net, x, y)
print("max relative error:", checker.max_relative_error(results))
```
