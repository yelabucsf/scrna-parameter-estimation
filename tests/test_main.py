import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from memento import main


def test_get_test_genes_and_indices_preserves_requested_order():
    adata = ad.AnnData(
        X=sparse.csr_matrix(np.zeros((2, 4))),
        var=pd.DataFrame(index=["g0", "g1", "g2", "g3"]),
    )
    treatment_for_gene = {"g3": ["tx"], "g1": ["tx"]}

    genes, indices = main._get_test_genes_and_indices(adata, treatment_for_gene)

    assert genes == ["g3", "g1"]
    np.testing.assert_array_equal(indices, [3, 1])


def test_get_test_genes_and_indices_reports_missing_genes():
    adata = ad.AnnData(
        X=sparse.csr_matrix(np.zeros((2, 1))),
        var=pd.DataFrame(index=["present"]),
    )

    with pytest.raises(ValueError, match="missing"):
        main._get_test_genes_and_indices(adata, {"missing": ["tx"]})


def test_ht_mean_uses_the_requested_gene_index(monkeypatch):
    adata = ad.AnnData(
        X=sparse.csr_matrix(np.zeros((4, 3))),
        var=pd.DataFrame(index=["g0", "g1", "g2"]),
    )
    groups = ["sg^sample_1", "sg^sample_2"]
    adata.uns["memento"] = {
        "groups": groups,
        "estimator_type": "hyper_relative",
        "1d_moments": {
            groups[0]: [np.array([1.0, 2.0, 30.0]), np.ones(3), np.ones(3)],
            groups[1]: [np.array([4.0, 5.0, 60.0]), np.ones(3), np.ones(3)],
        },
        "group_cells": {
            groups[0]: sparse.csc_matrix([[1, 2, 30], [4, 5, 60]]),
            groups[1]: sparse.csc_matrix([[7, 8, 90], [10, 11, 120]]),
        },
        "approx_size_factor": {group: np.ones(2) for group in groups},
        "mv_regressor": {group: np.ones(3) for group in groups},
        "group_q": {group: 0.1 for group in groups},
    }
    captured = []

    def capture_summary_statistics(**kwargs):
        captured.append(kwargs)
        return None

    monkeypatch.setattr(
        main.hypothesis_test,
        "_mean_summary_statistics",
        capture_summary_statistics,
    )

    treatment = pd.DataFrame({"tx": [0, 1]})
    result = main.ht_mean(
        adata,
        treatment=treatment,
        treatment_for_gene={"g2": ["tx"]},
        num_boot=3,
        num_cpus=1,
        verbose=0,
        return_stats=True,
    )

    assert result == [None]
    assert captured[0]["true_mean"] == [30.0, 60.0]
    np.testing.assert_array_equal(captured[0]["cells"][0].toarray().ravel(), [30, 60])


def test_parallel_task_seeds_are_reproducible_and_distinct():
    first = main._spawn_task_random_states(1234, 10)
    second = main._spawn_task_random_states(1234, 10)

    assert first == second
    assert len(set(first)) == len(first)
    assert main._spawn_task_random_states(None, 3) == [None, None, None]


def test_moment_pipeline_smoke_test():
    rng = np.random.default_rng(7)
    counts = rng.poisson(rng.uniform(0.5, 4.0, size=(80, 16)))
    obs = pd.DataFrame(
        {
            "capture_rate": np.full(80, 0.1),
            "condition": np.repeat(["control", "treated"], 40),
        },
        index=[f"cell_{idx}" for idx in range(80)],
    )
    var = pd.DataFrame(index=[f"gene_{idx}" for idx in range(16)])
    adata = ad.AnnData(X=sparse.csr_matrix(counts), obs=obs, var=var)

    main.setup_memento(
        adata,
        q_column="capture_rate",
        filter_mean_thresh=0.01,
        trim_percent=0.5,
        shrinkage=0.1,
        min_cell_count=10,
    )
    main.create_groups(adata, ["condition"])
    main.compute_1d_moments(adata, min_perc_group=0.0)

    means, variances, cell_counts = main.get_1d_moments(adata)
    assert means.shape == variances.shape
    assert means.shape[0] == adata.n_vars
    assert set(cell_counts.values()) == {40}


def test_get_groups_encodes_non_numeric_two_level_column():
    # Regression test for issue #61: a non-numeric, two-level label column
    # (e.g. 'control'/'stim') was previously left as raw strings by get_groups,
    # which downstream regression code (used by binary_test_1d/binary_test_2d)
    # cannot cast to float. It should now be auto-encoded as 0/1.
    adata = ad.AnnData(X=sparse.csr_matrix(np.zeros((1, 1))))
    adata.uns["memento"] = {
        "groups": ["sg^control", "sg^stim"],
        "label_delimiter": "^",
        "label_columns": ["treatment"],
    }

    result = main.get_groups(adata)

    assert pd.api.types.is_numeric_dtype(result["treatment"])
    assert result["treatment"].tolist() == [0.0, 1.0]


def test_get_groups_leaves_multi_level_non_numeric_column_unencoded():
    # A categorical column with more than two levels can't be meaningfully
    # collapsed into a single 0/1 treatment indicator, so get_groups should
    # leave it as-is rather than guessing an encoding.
    adata = ad.AnnData(X=sparse.csr_matrix(np.zeros((1, 1))))
    adata.uns["memento"] = {
        "groups": ["sg^a", "sg^b", "sg^c"],
        "label_delimiter": "^",
        "label_columns": ["treatment"],
    }

    result = main.get_groups(adata)

    assert result["treatment"].tolist() == ["a", "b", "c"]
    assert not pd.api.types.is_numeric_dtype(result["treatment"])


def test_get_groups_still_coerces_purely_numeric_column():
    # Existing behavior (columns that already look numeric, e.g. '0'/'1')
    # should be unaffected by the new binary-encoding fallback.
    adata = ad.AnnData(X=sparse.csr_matrix(np.zeros((1, 1))))
    adata.uns["memento"] = {
        "groups": ["sg^0", "sg^1"],
        "label_delimiter": "^",
        "label_columns": ["treatment"],
    }

    result = main.get_groups(adata)

    assert pd.api.types.is_numeric_dtype(result["treatment"])
    assert result["treatment"].tolist() == [0, 1]


def test_binary_test_1d_treatment_col_is_numeric_end_to_end():
    # End-to-end regression test for issue #61, reproducing the reported
    # usage: setup_memento -> create_groups -> get_groups with a non-numeric
    # treatment_col, exactly as binary_test_1d assembles its design matrix.
    rng = np.random.default_rng(0)
    counts = rng.poisson(2.0, size=(20, 4))
    obs = pd.DataFrame(
        {
            "capture_rate": np.full(20, 0.1),
            "condition": np.repeat(["control", "stim"], 10),
        },
        index=[f"cell_{idx}" for idx in range(20)],
    )
    var = pd.DataFrame(index=[f"gene_{idx}" for idx in range(4)])
    adata = ad.AnnData(X=sparse.csr_matrix(counts), obs=obs, var=var)

    main.setup_memento(adata, q_column="capture_rate", filter_mean_thresh=0.0)
    main.create_groups(adata, ["condition"])

    sample_meta = main.get_groups(adata)[["condition"]]

    assert pd.api.types.is_numeric_dtype(sample_meta["condition"])
    assert set(sample_meta["condition"].unique()) == {0.0, 1.0}
