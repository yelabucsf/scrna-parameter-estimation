"""Cluster (e.g. donor-level) replicate resampling: resample_rep=True with cluster=..."""
import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from memento import hypothesis_test, main


def test_cluster_assignments_keep_whole_clusters_together():
    cluster = np.array([0, 0, 1, 2, 2, 2, 3])
    rows, iterations, valid = hypothesis_test._cluster_assignments(
        cluster, num_boot=200, rng=np.random.default_rng(0))

    assert rows.shape == iterations.shape == valid.shape == (4 * 3, 201)
    # Column 0 is the observed data: every row once, cell-bootstrap column 0.
    np.testing.assert_array_equal(rows[valid[:, 0], 0], np.arange(7))
    assert (iterations[:, 0] == 0).all()
    assert iterations[:, 1:].min() >= 1 and iterations[:, 1:].max() <= 200

    for draw in range(1, 201):
        counts = np.bincount(rows[valid[:, draw], draw], minlength=7)
        # Rows of one cluster always appear the same number of times...
        for c in np.unique(cluster):
            assert len(set(counts[cluster == c])) == 1
        # ...and each draw holds exactly as many clusters as the data.
        assert sum(counts[cluster == c][0] for c in np.unique(cluster)) == 4
        # Padding repeats a row that is already part of the draw.
        assert set(rows[~valid[:, draw], draw]) <= set(rows[valid[:, draw], draw])


def _clustered_inputs(seed=0, n_donor=12, n_boot=400):
    # Two rows per donor that share a large donor effect; donor-level treatment.
    rng = np.random.default_rng(seed)
    donor = np.repeat(np.arange(n_donor), 2)
    treatment = (np.arange(n_donor) % 2).astype(float)[donor][:, None]
    covariate = np.tile([0.0, 1.0], n_donor)[:, None]
    nc = rng.integers(100, 800, donor.size).astype(float)
    boot = rng.normal(0, 0.15, n_donor)[donor][:, None] + rng.normal(0, 0.01, (donor.size, n_boot + 1))
    return covariate, treatment, boot, nc, donor


def test_cluster_resampling_widens_se_when_rows_share_a_donor_effect():
    covariate, treatment, boot, nc, donor = _clustered_inputs()
    kwargs = dict(resample_rep=True, rng=np.random.default_rng(1))
    _, se_rows, _ = hypothesis_test._regress_2d(covariate, treatment, boot.copy(), nc, **kwargs)
    kwargs["rng"] = np.random.default_rng(1)
    coef, se_cluster, pval = hypothesis_test._regress_2d(
        covariate, treatment, boot.copy(), nc, cluster=donor, **kwargs)

    assert np.isfinite([coef[0], se_cluster[0], pval[0]]).all()
    assert se_cluster[0] > 1.2 * se_rows[0]


def test_cluster_resampling_leaves_observed_coefficient_unchanged():
    covariate, treatment, boot, nc, donor = _clustered_inputs(seed=2)
    plain, _, _ = hypothesis_test._regress_2d(covariate, treatment, boot.copy(), nc)
    clustered, _, _ = hypothesis_test._regress_2d(
        covariate, treatment, boot.copy(), nc, resample_rep=True,
        rng=np.random.default_rng(3), cluster=donor)
    np.testing.assert_allclose(clustered, plain)


def test_cluster_resampling_1d_runs_with_unequal_cluster_sizes():
    rng = np.random.default_rng(4)
    donor = np.array([0, 0, 0, 1, 2, 2, 3, 4, 4, 5])
    treatment = (donor % 2).astype(float)[:, None]
    covariate = np.ones((donor.size, 1))
    nc = rng.integers(100, 800, donor.size).astype(float)
    log_mean = rng.normal(0, 0.2, 6)[donor][:, None] + rng.normal(0, 0.02, (donor.size, 201))
    out = hypothesis_test._regress_1d(
        covariate, treatment, log_mean.copy(), log_mean.copy() - 1, nc,
        resample_rep=True, rng=np.random.default_rng(5), cluster=donor)
    assert all(np.isfinite(v).all() for v in out)


def _donor_by_batch_adata(seed, n_donors=8, extra_labels=()):
    rng = np.random.default_rng(seed)
    n_cells = n_donors * 2 * 120
    obs = pd.DataFrame(
        {
            "capture_rate": 0.1,
            "donor": np.repeat([f"d{i}" for i in range(n_donors)], 240),
            "batch": np.tile(np.repeat(["b0", "b1"], 120), n_donors),
        },
        index=[f"c{i}" for i in range(n_cells)],
    )
    adata = ad.AnnData(
        X=sparse.csr_matrix(rng.poisson(5.0, size=(n_cells, 30)).astype(float)),
        obs=obs, var=pd.DataFrame(index=[f"gene_{i}" for i in range(30)]))
    for label in extra_labels:
        adata.obs[label] = "only"
    main.setup_memento(adata, q_column="capture_rate", filter_mean_thresh=0.0)
    main.create_groups(adata, ["donor", "batch", *extra_labels])
    main.compute_1d_moments(adata, min_perc_group=0.3)
    meta = main.get_groups(adata)
    treatment = pd.DataFrame({"sex": (meta["donor"].str[1:].astype(int) % 2).astype(float)})
    covariate = pd.DataFrame({"batch": meta["batch"].astype(float)})
    return adata, treatment, covariate


def test_cluster_resampling_end_to_end_1d_and_2d():
    adata, treatment, covariate = _donor_by_batch_adata(30)
    genes = adata.uns["memento"]["gene_list"]
    main.compute_2d_moments(adata, [(genes[0], genes[1]), (genes[2], genes[3])])

    main.ht_1d_moments(adata, treatment=treatment, covariate=covariate, num_boot=200,
                       verbose=0, resample_rep=True, cluster="donor")
    result_1d = main.get_1d_ht_result(adata)
    assert len(result_1d) == adata.shape[1]
    assert np.isfinite(result_1d[["de_coef", "de_se", "de_pval"]].to_numpy()).all()

    main.ht_2d_moments(adata, treatment=treatment, covariate=covariate, num_boot=200,
                       verbose=0, resample_rep=True, cluster="donor")
    result_2d = main.get_2d_ht_result(adata)
    assert len(result_2d) == 2
    assert np.isfinite(result_2d[["corr_coef", "corr_se", "corr_pval"]].to_numpy()).all()


def test_cluster_resampling_is_seeded():
    adata, treatment, covariate = _donor_by_batch_adata(31)

    def run():
        main.ht_1d_moments(adata, treatment=treatment, covariate=covariate, num_boot=100,
                           verbose=0, resample_rep=True, cluster="donor", random_state=7)
        return main.get_1d_ht_result(adata)

    pd.testing.assert_frame_equal(run(), run())


@pytest.mark.parametrize("options, error, match", [
    (dict(cluster="donor"), ValueError, "resample_rep=True"),
    (dict(cluster="individual", resample_rep=True), ValueError, "label_columns"),
])
def test_cluster_option_is_validated(options, error, match):
    adata, treatment, covariate = _donor_by_batch_adata(32, n_donors=4)
    with pytest.raises(error, match=match):
        main.ht_1d_moments(adata, treatment=treatment, covariate=covariate, num_boot=50,
                           verbose=0, **options)


def test_cluster_needs_at_least_two_clusters():
    adata, treatment, covariate = _donor_by_batch_adata(33, n_donors=4, extra_labels=("site",))
    with pytest.raises(ValueError, match="at least two clusters"):
        main.ht_1d_moments(adata, treatment=treatment, covariate=covariate, num_boot=50,
                           verbose=0, resample_rep=True, cluster="site")
