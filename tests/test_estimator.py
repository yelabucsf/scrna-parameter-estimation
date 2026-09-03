import numpy as np
import pytest

from memento import estimator


def test_fit_mv_regressor_works_with_enough_valid_genes():
    # Sanity check: normal input is unaffected by the new guard.
    rng = np.random.default_rng(0)
    mean = rng.uniform(0.1, 5.0, size=50)
    var = mean * rng.uniform(1.0, 3.0, size=50)  # roughly overdispersed

    poly = estimator._fit_mv_regressor(mean, var)

    assert poly.shape == (3,)  # degree-2 polyfit returns 3 coefficients


def test_fit_mv_regressor_raises_informative_error_on_empty_input():
    # Regression test for issue #29: a group with no genes passing
    # (mean > 0) & (var > 0) -- e.g. a donor with too few cells or too low
    # sequencing depth -- previously crashed deep inside np.polyfit with an
    # opaque "TypeError: expected non-empty vector for x". It should now
    # raise a clear, actionable error instead.
    mean = np.zeros(20)
    var = np.zeros(20)

    with pytest.raises(ValueError, match="No genes with a positive mean"):
        estimator._fit_mv_regressor(mean, var)


def test_fit_mv_regressor_error_names_the_offending_group():
    # The error should identify which group failed, since this is raised
    # from within a per-group loop (compute_1d_moments) where the caller
    # needs to know which group to investigate.
    mean = np.zeros(20)
    var = np.zeros(20)

    with pytest.raises(ValueError, match="sg\\^C"):
        estimator._fit_mv_regressor(mean, var, group="sg^C")


def test_fit_mv_regressor_tolerates_few_but_nonzero_valid_genes():
    # np.polyfit itself tolerates 1-2 points for a degree-2 fit (producing
    # an underdetermined-but-numeric result), so the new guard should only
    # trigger on genuinely empty input, not merely small input -- this
    # preserves existing (if statistically weak) behavior for edge cases
    # that previously "worked".
    mean = np.array([1.0, 2.0])
    var = np.array([1.5, 3.0])

    poly = estimator._fit_mv_regressor(mean, var)

    assert poly.shape == (3,)
