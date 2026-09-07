"""The trimmed size-factor estimator the publication scripts call as
``RNAHypergeometric.estimate_size_factor``.

That method was dropped in the object-oriented rewrite, but the same computation
still lives in this repository's ``memento.main.setup_memento``: fit a
mean-variance regressor over all cells, keep the least residually-variable genes,
and size-factor off those genes only. The module is loaded by file path because
the OO rewrite occupies the ``memento`` package name at import time.
"""

import importlib.util
import os

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_spec = importlib.util.spec_from_file_location(
    'memento_legacy_estimator', os.path.join(_REPO_ROOT, 'memento', 'estimator.py'))
_legacy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_legacy)


def trimmed_size_factor(data, q, shrinkage=0.6, filter_mean_thresh=0.07, trim_percent=0.05):
    naive_size_factor = _legacy._estimate_size_factor(
        data, 'hyper_relative', total=True, shrinkage=0.0)

    all_m, all_v = _legacy._hyper_1d_relative(
        data=data, n_obs=data.shape[0], q=q, size_factor=naive_size_factor)
    all_m[data.mean(axis=0).A1 < filter_mean_thresh] = 0

    residual_variance = _legacy._residual_variance(
        all_m, all_v, _legacy._fit_mv_regressor(all_m, all_v))
    upper_limit = np.quantile(residual_variance[np.isfinite(residual_variance)], trim_percent)
    residual_variance[~np.isfinite(residual_variance)] = np.inf
    mask = residual_variance <= upper_limit

    return _legacy._estimate_size_factor(data, 'hyper_relative', mask=mask, shrinkage=shrinkage)
