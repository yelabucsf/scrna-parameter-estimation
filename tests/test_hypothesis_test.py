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


def test_cross_coef_handles_zero_variance_column_without_inf_or_crash():
    # Regression test for PR #16 (nkschaefer): a constant column in A (e.g. a
    # treatment/covariate with no variation across samples) previously caused
    # a division-by-zero in _cross_coef, propagating inf/nan into results.
    # The coefficient for such a column is genuinely undefined, so it comes
    # out as NaN (not 0) -- this keeps coef/se/pval consistent downstream in
    # _regress_2d, since _compute_asl already treats a degenerate row as an
    # invalid test.
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
    assert not np.isinf(result).any()
    # The constant column's row is undefined -> NaN, not silently zeroed.
    assert np.all(np.isnan(result[0]))
    # The non-constant column should still produce a real (nonzero) estimate.
    assert not np.isnan(result[1]).any()
    assert not np.allclose(result[1], 0)


def test_compute_asl_returns_nan_for_degenerate_row_from_cross_coef():
    # Regression test tying together PR #16's fix: a zero-variance treatment
    # column now produces an all-NaN row from _cross_coef. _compute_asl's
    # original guard (checking perm_diff == perm_diff.mean()) doesn't catch
    # an all-NaN array, since NaN != NaN -- it must explicitly check for it.
    all_nan_row = np.full(21, np.nan)

    with np.errstate(invalid="raise"):
        pval = hypothesis_test._compute_asl(all_nan_row)

    assert np.isnan(pval)


def test_compute_asl_still_handles_all_equal_non_nan_row():
    # Sanity check that the original all-equal guard (e.g. a row of all
    # zeros, which _cross_coef could still legitimately produce for reasons
    # unrelated to the zero-variance guard) is unaffected.
    all_zero_row = np.zeros(21)

    pval = hypothesis_test._compute_asl(all_zero_row)

    assert np.isnan(pval)


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


def test_cross_coef_resampled_handles_zero_variance_iteration_without_warning():
    # Regression test for issue #38: unlike a treatment column that is
    # constant across the whole dataset (filtered out upstream before this
    # function is ever called), a single bootstrap iteration can, purely by
    # chance, resample a subset for which an otherwise-varying column ends
    # up with zero variance. The resulting coefficient is genuinely
    # undefined for that one iteration (fine, tolerated downstream), but it
    # should come out as a clean NaN rather than triggering a
    # RuntimeWarning from an actual division by zero.
    rng = np.random.default_rng(0)
    num_rep, num_boot, n_treatment_cols = 5, 4, 1

    A = rng.normal(size=(num_rep, num_boot, n_treatment_cols))
    A[:, 1, :] = 7.0  # bootstrap iteration 1 is degenerate: zero variance
    B = rng.normal(size=(num_rep, num_boot))
    sample_weight = np.ones((num_rep, num_boot))

    with np.errstate(divide="raise", invalid="raise"):
        beta = hypothesis_test._cross_coef_resampled(A, B, sample_weight)

    assert beta.shape == (n_treatment_cols, num_boot)
    assert np.all(np.isnan(beta[:, 1]))
    # Other iterations, with genuine variance, are unaffected.
    assert not np.isnan(beta[:, 0]).any()
    assert not np.isnan(beta[:, 2]).any()
    assert not np.isnan(beta[:, 3]).any()


def test_replicate_assignment_count_and_observed_column():
    groups,draws=hypothesis_test._replicate_assignments(7,101,np.random.default_rng(4))
    assert groups.shape==draws.shape==(7,102)
    np.testing.assert_array_equal(groups[:,0],np.arange(7))
    np.testing.assert_array_equal(draws[:,0],0)
    assert (draws[:,1:]>=1).all() and (draws[:,1:]<=101).all()


def test_resampled_identical_treatments_are_nan_despite_roundoff():
    a=np.full((3,20,1),.1)
    b=np.random.default_rng(5).normal(size=(3,20))
    weights=np.broadcast_to(np.array([7.,11.,13.])[:,None],(3,20))
    got=hypothesis_test._cross_coef_resampled(a,b,weights)
    assert np.isnan(got).all()


def test_regression_empty_draws_return_correct_output_shape():
    cov=np.ones((3,1));tx=np.array([[0.],[1.],[2.]])
    y=np.full((3,4),np.nan)
    one=hypothesis_test._regress_1d(cov,tx,y,y,np.ones(3),resample_rep=True)
    two=hypothesis_test._regress_2d(cov,tx,y,np.ones(3),resample_rep=True)
    assert np.shape(one)==(6,1) and np.shape(two)==(3,1)
    assert np.isnan(one).all() and np.isnan(two).all()
    assert np.isnan(hypothesis_test._compute_asl(np.array([1.,np.nan,np.nan])))


def test_regress_2d_one_sample_is_weighted_mean_of_group_correlations():
    # One-sample test (H0: correlation = 0): the coefficient is the
    # cell-count-weighted mean of the observed group correlations, and the
    # bootstrap null is the same weighted mean over each bootstrap column.
    # Covariates and treatment are ignored by construction.
    rng = np.random.default_rng(11)
    n_groups, num_boot = 5, 400
    boot_corr = np.clip(rng.normal(0.3, 0.05, size=(n_groups, num_boot + 1)), -0.99, 0.99)
    Nc_list = np.array([50, 120, 80, 300, 60])
    covariate = np.ones((n_groups, 1))
    treatment = np.ones((n_groups, 1))

    coef, se, pval = hypothesis_test._regress_2d(
        covariate, treatment, boot_corr, Nc_list, one_sample=True)

    weighted = np.average(boot_corr, axis=0, weights=Nc_list)
    np.testing.assert_allclose(coef, [weighted[0]])
    np.testing.assert_allclose(se, [np.std(weighted[1:])])
    assert pval.shape == (1,) and pval[0] < 1e-6


def test_regress_2d_one_sample_null_correlation_is_not_significant():
    rng = np.random.default_rng(12)
    boot_corr = rng.normal(0.0, 0.05, size=(6, 1001))
    boot_corr[:, 0] = 0.0
    Nc_list = np.full(6, 100)

    coef, _, pval = hypothesis_test._regress_2d(
        np.ones((6, 1)), np.ones((6, 1)), boot_corr, Nc_list, one_sample=True)

    assert abs(coef[0]) < 1e-12
    assert pval[0] > 0.5


def _heterogeneous_group_correlations(rng, n_groups=12, num_boot=2000):
    # Group correlations that vary with a continuous group-level covariate
    # (e.g. donor age), plus small cell-bootstrap noise around each group.
    age = rng.uniform(20, 80, n_groups)
    centre = 0.1 + 0.005 * (age - 50)
    boot_corr = centre[:, None] + rng.normal(0, 0.01, size=(n_groups, num_boot + 1))
    Nc_list = rng.integers(50, 300, n_groups)
    return boot_corr, Nc_list, age[:, None]


def test_regress_2d_one_sample_covariate_is_a_no_op_without_resampling():
    # The weighted mean of a weighted least-squares fit with an intercept is
    # the weighted mean of the data, so covariate adjustment cannot move the
    # one-sample statistic unless groups are resampled.
    boot_corr, Nc_list, age = _heterogeneous_group_correlations(np.random.default_rng(21))
    ones = np.ones_like(age)

    plain = hypothesis_test._regress_2d(ones, ones, boot_corr, Nc_list, one_sample=True)
    adjusted = hypothesis_test._regress_2d(age, ones, boot_corr, Nc_list, one_sample=True)

    np.testing.assert_allclose(adjusted, plain)


def test_regress_2d_one_sample_resample_rep_residualizes_on_covariates():
    boot_corr, Nc_list, age = _heterogeneous_group_correlations(np.random.default_rng(22))
    ones = np.ones_like(age)

    def run(covariate, resample_rep):
        return hypothesis_test._regress_2d(
            covariate, ones, boot_corr, Nc_list, resample_rep=resample_rep,
            rng=np.random.default_rng(5), one_sample=True)

    cells_only = run(ones, False)
    groups_plain = run(ones, True)
    groups_adjusted = run(age, True)

    # The observed coefficient is the weighted mean on every path.
    weighted = np.average(boot_corr[:, 0], weights=Nc_list)
    for coef, _, _ in (cells_only, groups_plain, groups_adjusted):
        np.testing.assert_allclose(coef, [weighted])

    # Resampling groups adds the between-group spread of the correlations;
    # adjusting for the covariate that drives that spread removes it again.
    assert groups_plain[1][0] > 5 * cells_only[1][0]
    assert groups_adjusted[1][0] < 0.5 * groups_plain[1][0]


def test_regress_2d_one_sample_resample_rep_matches_manual_resampling():
    boot_corr, Nc_list, age = _heterogeneous_group_correlations(np.random.default_rng(23), num_boot=300)
    ones = np.ones_like(age)

    coef, se, _ = hypothesis_test._regress_2d(
        age, ones, boot_corr, Nc_list, resample_rep=True,
        rng=np.random.default_rng(9), one_sample=True)

    X = np.column_stack([np.ones(len(age)), age[:, 0]])
    W = np.diag(Nc_list.astype(float))
    beta = np.linalg.solve(X.T @ W @ X, X.T @ W @ boot_corr)
    adjusted = boot_corr - X @ beta + np.average(boot_corr, axis=0, weights=Nc_list)
    rows, draws = hypothesis_test._replicate_assignments(len(age), 300, np.random.default_rng(9))
    expected = np.average(adjusted[rows, draws], axis=0, weights=Nc_list[rows])

    np.testing.assert_allclose(coef, [expected[0]])
    np.testing.assert_allclose(se, [np.std(expected[1:])])


def test_resampled_treatments_equal_up_to_rounding_are_nan():
    # Distinct rows whose residualized treatment differs only in the last
    # bits (seen with a binary treatment residualized on a covariate) used
    # to pass the exact-equality guard and produce slopes around 1e14.
    value = -5 / 11
    nearly_equal = np.array([value, np.nextafter(value, 0), value, np.nextafter(value, 0)])
    contrast = np.array([value, 6 / 11, value, 6 / 11])
    A = np.stack([contrast, nearly_equal], axis=1)[:, :, None]      # (rows, draws, columns)
    B = np.random.default_rng(0).normal(size=(4, 2))
    weights = np.full((4, 2), 150.0)

    beta = hypothesis_test._cross_coef_resampled(A, B, weights)

    assert np.isfinite(beta[0, 0])
    assert np.isnan(beta[0, 1])
