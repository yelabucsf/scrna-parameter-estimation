"""Convenience wrappers (binary_test_1d / binary_test_2d)."""
import warnings

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

import memento


def _binary_wrapper_adata(seed):
    rng = np.random.default_rng(seed)
    n_cells = 4 * 150
    obs = pd.DataFrame(
        {
            "stim": np.tile(np.repeat(["ctrl", "stim"], 150), 2),
            "donor": np.repeat(["d0", "d1"], 300),
        },
        index=[f"c{i}" for i in range(n_cells)],
    )
    return ad.AnnData(
        X=sparse.csr_matrix(rng.poisson(5.0, size=(n_cells, 30)).astype(float)),
        obs=obs, var=pd.DataFrame(index=[f"gene_{i}" for i in range(30)]))


def test_binary_test_1d_replicates_is_deprecated_but_still_groups():
    adata = _binary_wrapper_adata(40)
    with pytest.warns(FutureWarning, match="'replicates' argument"):
        with_replicates = memento.binary_test_1d(
            adata, capture_rate=0.1, treatment_col="stim", num_cpus=1, num_boot=100,
            verbose=0, replicates=["donor"])
    assert len(with_replicates) > 0
    assert set(with_replicates["tx"]) == {"stim"}


def test_binary_test_1d_without_replicates_does_not_warn():
    adata = _binary_wrapper_adata(41)
    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=FutureWarning, message=".*replicates.*")
        result = memento.binary_test_1d(
            adata, capture_rate=0.1, treatment_col="stim", num_cpus=1, num_boot=100, verbose=0)
    assert len(result) > 0
