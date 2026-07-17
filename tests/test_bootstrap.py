from collections import Counter

import numpy as np
import pytest
from scipy import sparse

from memento import bootstrap, estimator


def _state_counts(expression, size_factor):
    return Counter(
        (float(sf), *row)
        for sf, row in zip(size_factor, np.asarray(expression).tolist())
    )


def test_unique_expr_compresses_exact_states_without_touching_global_rng():
    expression = np.array(
        [[0, 1], [0, 1], [2, 0], [2, 0], [2, 0], [4, 3]], dtype=np.int64
    )
    size_factor = np.array([100.0, 100.0, 100.0, 200.0, 200.0, 100.0])

    np.random.seed(17)
    expected_next_random = np.random.random()
    np.random.seed(17)
    inv_sf, inv_sf_sq, unique_expression, counts = bootstrap._unique_expr(
        sparse.csc_matrix(expression), size_factor
    )
    actual_next_random = np.random.random()

    actual = Counter()
    for inv, row, count in zip(inv_sf.ravel(), unique_expression.tolist(), counts):
        actual[(float(1 / inv), *row)] = int(count)
    assert actual == _state_counts(expression, size_factor)
    np.testing.assert_allclose(inv_sf_sq, inv_sf**2)
    assert actual_next_random == expected_next_random


def test_unique_expr_supports_non_integer_custom_estimator_input():
    expression = np.array([[0.5], [0.5], [1.25], [1.25]])
    size_factor = np.array([10.0, 10.0, 10.0, 20.0])

    inv_sf, _, unique_expression, counts = bootstrap._unique_expr(
        sparse.csc_matrix(expression), size_factor
    )

    actual = Counter()
    for inv, row, count in zip(inv_sf.ravel(), unique_expression.tolist(), counts):
        actual[(float(1 / inv), *row)] = int(count)
    assert actual == _state_counts(expression, size_factor)


def test_bootstrap_batching_preserves_seeded_results():
    rng = np.random.default_rng(123)
    expression = sparse.csc_matrix(rng.poisson(1.5, size=(500, 1)))
    size_factor = rng.choice([500.0, 1000.0, 2000.0], size=500)

    unbatched = bootstrap._bootstrap_1d(
        expression,
        size_factor,
        q=0.1,
        _estimator_1d=estimator._hyper_1d_relative,
        num_boot=101,
        batch_size=101,
    )
    batched = bootstrap._bootstrap_1d(
        expression,
        size_factor,
        q=0.1,
        _estimator_1d=estimator._hyper_1d_relative,
        num_boot=101,
        batch_size=7,
    )

    np.testing.assert_allclose(batched[0], unbatched[0], rtol=0, atol=0)
    np.testing.assert_allclose(batched[1], unbatched[1], rtol=0, atol=0)


def test_two_dimensional_bootstrap_batching_preserves_seeded_results():
    rng = np.random.default_rng(321)
    expression = sparse.csc_matrix(rng.poisson(1.5, size=(500, 2)))
    size_factor = rng.choice([500.0, 1000.0, 2000.0], size=500)

    arguments = {
        "data": expression,
        "size_factor": size_factor,
        "q": 0.1,
        "_estimator_1d": estimator._hyper_1d_relative,
        "_estimator_cov": estimator._hyper_cov_relative,
        "num_boot": 101,
    }
    unbatched = bootstrap._bootstrap_2d(**arguments, batch_size=101)
    batched = bootstrap._bootstrap_2d(**arguments, batch_size=7)

    for batched_moment, unbatched_moment in zip(batched, unbatched):
        np.testing.assert_allclose(batched_moment, unbatched_moment, rtol=0, atol=0)


def test_compressed_and_sparse_estimators_are_equivalent():
    expression = np.array([[0, 1], [2, 0], [2, 0], [4, 3], [0, 1]])
    matrix = sparse.csc_matrix(expression)
    size_factor = np.array([100.0, 200.0, 200.0, 100.0, 100.0])
    inv_sf, inv_sf_sq, unique_expression, counts = bootstrap._unique_expr(
        matrix, size_factor
    )
    frequencies = counts[:, np.newaxis]

    sparse_mean, sparse_var = estimator._hyper_1d_relative(
        matrix, matrix.shape[0], q=0.1, size_factor=size_factor
    )
    compressed_mean, compressed_var = estimator._hyper_1d_relative(
        (unique_expression, frequencies),
        matrix.shape[0],
        q=0.1,
        size_factor=(inv_sf, inv_sf_sq),
    )
    np.testing.assert_allclose(compressed_mean, sparse_mean)
    np.testing.assert_allclose(compressed_var, sparse_var)

    sparse_cov = estimator._hyper_cov_relative(
        matrix,
        matrix.shape[0],
        size_factor,
        q=0.1,
        idx1=np.array([0]),
        idx2=np.array([1]),
    )
    compressed_cov = estimator._hyper_cov_relative(
        (unique_expression[:, 0:1], unique_expression[:, 1:2], frequencies),
        matrix.shape[0],
        (inv_sf, inv_sf_sq),
        q=0.1,
    )
    np.testing.assert_allclose(compressed_cov, sparse_cov)


def test_poisson_estimator_names_resolve_to_the_implemented_functions():
    assert estimator._get_estimator_1d("poi_relative") is estimator._poisson_1d_relative
    assert estimator._get_estimator_cov("poi_relative") is estimator._poisson_cov_relative
    with pytest.raises(ValueError, match="Unknown estimator_type"):
        estimator._get_estimator_1d("not_an_estimator")
