import numpy as np
import pandas as pd

from memento import hypothesis_test


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
