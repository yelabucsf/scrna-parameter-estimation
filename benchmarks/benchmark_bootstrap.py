"""Small, reproducible benchmark for bootstrap state compression."""

from time import perf_counter

import numpy as np
from scipy import sparse

from memento import bootstrap


def legacy_random_projection(expression, size_factor):
    """The pre-refactor compression path, retained only for comparison."""

    generator = np.random.default_rng(42)
    code = expression.dot(generator.random(expression.shape[1]))
    code += generator.random() * size_factor
    _, index, counts = np.unique(code, return_index=True, return_counts=True)
    return expression[index], size_factor[index], counts


def best_time(function, repeats=10):
    timings = []
    for _ in range(repeats):
        start = perf_counter()
        function()
        timings.append(perf_counter() - start)
    return min(timings)


def main():
    rng = np.random.default_rng(42)
    num_cells = 500_000
    expression = sparse.csc_matrix(rng.negative_binomial(2, 0.7, (num_cells, 2)))
    size_factor = rng.choice(np.linspace(500, 5000, 30), size=num_cells)

    legacy_seconds = best_time(lambda: legacy_random_projection(expression, size_factor))
    exact_seconds = best_time(lambda: bootstrap._unique_expr(expression, size_factor))

    print(f"cells={num_cells:,} genes={expression.shape[1]}")
    print(f"legacy_random_projection_seconds={legacy_seconds:.6f}")
    print(f"exact_mixed_radix_seconds={exact_seconds:.6f}")
    print(f"speedup={legacy_seconds / exact_seconds:.2f}x")


if __name__ == "__main__":
    main()
