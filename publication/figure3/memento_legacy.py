"""Readers for hypothesis-test results written by memento 0.0.6 / 0.0.9.

Every h5ad under hbec/binary_test_* was produced by those versions, whose
`uns['memento']['1d_ht']` layout differs from the current package's. Calling today's
`memento.get_1d_ht_result` on one raises KeyError: 'test_genes'.

The stored layout keeps one array per statistic, aligned to `adata.var.index`:

    mean_coef, mean_se, mean_asl   ->  de_coef, de_se, de_pval
    var_coef,  var_se,  var_asl    ->  dv_coef, dv_se, dv_pval

("asl" is the achieved significance level, i.e. the bootstrap p-value.) The renaming
here reproduces the column names the original notebooks consumed, so the analysis code
reads the same as it did then.
"""

import numpy as np
import pandas as pd
import scanpy as sc

import config

config.add_repo_to_path()
from memento.util import _fdrcorrect  # noqa: E402


def read_1d_ht(adata_or_path, add_fdr=True):
    """Return the 1D differential mean / variability table for one stored test."""
    adata = sc.read(adata_or_path) if isinstance(adata_or_path, str) else adata_or_path
    ht = adata.uns['memento']['1d_ht']

    if len(ht['mean_coef']) != adata.shape[1]:
        raise ValueError(
            f'{len(ht["mean_coef"])} statistics for {adata.shape[1]} genes; '
            'the stored results are not aligned to adata.var')

    result = pd.DataFrame({
        'gene': adata.var.index,
        'de_coef': ht['mean_coef'], 'de_se': ht['mean_se'], 'de_pval': ht['mean_asl'],
        'dv_coef': ht['var_coef'], 'dv_se': ht['var_se'], 'dv_pval': ht['var_asl'],
    })
    if add_fdr:
        result['de_fdr'] = safe_fdr(result['de_pval'].values)
        result['dv_fdr'] = safe_fdr(result['dv_pval'].values)
    return result


def safe_fdr(pvals):
    """FDR-correct, leaving non-finite p-values as 1 rather than dropping the gene."""
    out = np.ones(pvals.shape[0])
    finite = np.isfinite(pvals)
    if finite.sum():
        out[finite] = _fdrcorrect(pvals[finite])
    return out


MOMENT_NAMES = ['mean', 'variance', 'residual_variance']


def read_1d_moments(adata_or_path):
    """Per-group moment estimates stored alongside the test, in long form.

    `uns['memento']['1d_moments']` is a dict of group label -> (3, n_genes) array whose
    rows are mean, variance and residual variance. The original notebooks reached for
    the old `get_1d_moments(...)[1]` -- the variability table -- which is the
    `residual_variance` column here.
    """
    adata = sc.read(adata_or_path) if isinstance(adata_or_path, str) else adata_or_path
    stored = adata.uns['memento']['1d_moments']

    frames = []
    for group, values in stored.items():
        frame = pd.DataFrame(dict(zip(MOMENT_NAMES, values)))
        frame.insert(0, 'gene', adata.var.index)
        frame.insert(1, 'group', group)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)
