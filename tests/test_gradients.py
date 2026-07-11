"""
Regression test locking in the result of numerical gradient checking.

This does NOT re-derive the gradient check from scratch each time -
it just asserts that backprop's analytic gradients still agree with
the independent numerical estimate, on every run. If someone edits
Layer.compute_gradients() or NeuralNetwork.backprop() later and
introduces a sign error or a wrong transpose, this test catches it
immediately instead of it silently degrading training quality.
"""

from digit_classifier.diagnostics.gradient_checker import GradientChecker


def test_backprop_matches_numerical_gradient(small_network, single_example):
    x, y = single_example
    checker = GradientChecker(epsilon=1e-4, relative_error_threshold=1e-7)

    # Network is tiny (26 total parameters), so check all of them -
    # exhaustive is cheap here. Larger networks would use sample_size.
    results = checker.check(small_network, x, y, sample_size=None)

    assert len(results) > 0
    max_error = checker.max_relative_error(results)
    assert checker.passed(results), (
        f"Backprop gradient diverges from numerical estimate: "
        f"max relative error {max_error:.2e} exceeds threshold "
        f"{checker.relative_error_threshold:.2e}"
    )
