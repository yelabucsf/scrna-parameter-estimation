import numpy as np
import pandas as pd
from scipy import sparse

from memento import hypothesis_test


def test_influence_kurtosis_allocates_more_draws_to_heavy_tails():
    rng = np.random.default_rng(4)
    size_factor = np.ones(200)
    regular = sparse.csc_matrix(rng.poisson(2, size=(200, 1)))
    heavy = np.zeros((200, 1))
    heavy[0] = 100
    heavy = sparse.csc_matrix(heavy)

    regular_draws, _ = hypothesis_test._plan_bootstrap_draws(
        regular, size_factor, 0.04, 250, 2_000, 50
    )
    heavy_draws, _ = hypothesis_test._plan_bootstrap_draws(
        heavy, size_factor, 0.04, 250, 2_000, 50
    )

    assert regular_draws >= 250
    assert heavy_draws > regular_draws


def test_adaptive_mean_bootstrap_uses_preplanned_draw_count(monkeypatch):
    calls = []

    def stable_bootstrap(num_boot, **kwargs):
        calls.append(num_boot)
        values = np.tile(np.array([0.9, 1.0, 1.1, 1.0]), num_boot // 4 + 1)
        return values[:num_boot], np.full(num_boot, 10.0)

    def pseudobulk(*args, **kwargs):
        raise AssertionError("the estimator is called only by the bootstrap")

    pseudobulk.__name__ = "_pseudobulk"
    monkeypatch.setattr(
        hypothesis_test,
        "_plan_bootstrap_draws",
        lambda **kwargs: (250, 0.04),
    )
    monkeypatch.setattr(hypothesis_test.bootstrap, "_bootstrap_1d", stable_bootstrap)
    result = hypothesis_test._adaptive_mean_summary_statistics(
        true_mean=[1.0],
        true_res_var=[1.0],
        cells=[None],
        approx_sf=[None],
        num_boot=2_000,
        q=[0.1],
        _estimator_1d=pseudobulk,
        rng=np.random.default_rng(1),
        target_se_rse=0.05,
        min_boot=250,
        bootstrap_resolution=250,
    )

    assert calls == [250]
    np.testing.assert_array_equal(result[4], [250])


def test_cross_coef_matches_weighted_diagonal_reference():
    rng = np.random.default_rng(11)
    a = rng.normal(size=(25, 3))
    b = rng.normal(size=(25, 17))
    weights = rng.uniform(1, 100, size=25)

    centered_a = a - np.average(a, axis=0, weights=weights)
    centered_b = b - np.average(b, axis=0, weights=weights)
    sum_squares = np.average(centered_a**2, axis=0, weights=weights)
    expected = (
        centered_a.T.dot(np.diag(weights)).dot(centered_b)
        / weights.sum()
        / sum_squares[:, np.newaxis]
    )

    actual = hypothesis_test._cross_coef(a, b, weights)

    np.testing.assert_allclose(actual, expected)


def test_quasiml_honors_per_gene_treatments_and_covariates(monkeypatch):
    def capture_fit(**kwargs):
        return kwargs

    monkeypatch.setattr(hypothesis_test.util, "fit_loglinear", capture_fit)
    groups = ["group_0", "group_1", "group_2", "group_3"]
    genes = ["gene_0", "gene_1"]
    results = [
        (np.full(4, 0.1), np.full(4, 0.01)),
        (np.full(4, 0.2), np.full(4, 0.02)),
    ]
    treatment = pd.DataFrame(
        {"tx_0": [0, 0, 1, 1], "tx_1": [0, 1, 0, 1]}, index=groups
    )
    covariate = pd.DataFrame(
        {"intercept": np.ones(4), "batch": [0, 1, 0, 1]}, index=groups
    )

    fits, _ = hypothesis_test._ht_mean_quasiML(
        results=results,
        treatment=treatment,
        covariate=covariate,
        total_umi=np.full((4, 1), 1000.0),
        umi_depth=1000.0,
        group_names=groups,
        gene_names=genes,
        return_fits=True,
        num_cpus=1,
        treatment_for_gene={"gene_0": ["tx_0"], "gene_1": ["tx_1"]},
        covariate_for_gene={"gene_0": ["intercept"], "gene_1": ["intercept", "batch"]},
    )

    assert [(fit["gene"], fit["t"]) for fit in fits] == [
        ("gene_0", "tx_0"),
        ("gene_1", "tx_1"),
    ]
    assert fits[0]["exog"].columns.tolist() == ["intercept", "tx_0"]
    assert fits[1]["exog"].columns.tolist() == ["intercept", "batch", "tx_1"]


def test_cross_coef_handles_zero_variance_column_without_nan_or_inf():
    # Regression test for PR #16 (nkschaefer): a constant column in A (e.g. a
    # treatment/covariate with no variation across samples) previously caused
    # a division-by-zero in _cross_coef, propagating inf/nan into results.
    rng = np.random.default_rng(0)
    n = 50

    A = np.column_stack(
        [
            np.ones(n),  # constant column -> zero variance
            rng.normal(size=n),
        ]
    )
    B = rng.normal(size=(n, 3))
    weights = np.ones(n)

    with np.errstate(divide="raise", invalid="raise"):
        result = hypothesis_test._cross_coef(A, B, weights)

    assert result.shape == (2, 3)
    assert not np.isnan(result).any()
    assert not np.isinf(result).any()
    # The constant column's row should be zeroed out rather than blown up.
    np.testing.assert_array_equal(result[0], np.zeros(3))
    # The non-constant column should still produce a real (nonzero) estimate.
    assert not np.allclose(result[1], 0)


def test_cross_coef_matches_manual_calculation_when_no_zero_variance():
    # Sanity check that the zero-variance guard doesn't change results when
    # it isn't needed.
    rng = np.random.default_rng(1)
    n = 40

    A = rng.normal(size=(n, 2))
    B = rng.normal(size=(n, 2))
    weights = np.ones(n)

    result = hypothesis_test._cross_coef(A, B, weights)

    A_mA = A - A.mean(axis=0)
    B_mB = B - B.mean(axis=0)
    ssA = (A_mA**2).mean(axis=0)
    expected = (A_mA.T @ B_mB) / n / ssA[:, None]

    np.testing.assert_allclose(result, expected)
