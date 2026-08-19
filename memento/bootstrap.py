"""
    bootstrap.py
    
    This file contains functions for fast bootstraping.
"""

import numpy as np
import pandas as pd
import time


_MAX_DENSE_STATE_BINS = 1_000_000
_DEFAULT_BOOTSTRAP_BATCH_BYTES = 64 * 1024**2


def numpy_fill(arr):
    nan_idxs = np.isnan(arr)
    arr[nan_idxs] = np.nanmedian(arr)
    

def _convert_params(mu, alpha):
    """ 
    Convert mean/dispersion parameterization of a negative binomial to the ones scipy supports

    Parameters
    ----------
    mu : float 
       Mean of NB distribution.
    alpha : float
       Overdispersion parameter used for variance calculation.

    See https://en.wikipedia.org/wiki/Negative_binomial_distribution#Alternative_formulations
    """
    var = mu + alpha * mu ** 2
    p = mu / var
    r = mu ** 2 / (var - mu)
    return r, p
    

def _unique_expr(expr, size_factor):
    """
    Compress cells with identical expression and size factor states.

    ``expr`` is a sparse count matrix with one or two columns. Memento bins
    size factors before bootstrapping, so the number of distinct states is
    generally small even for very large cell populations.

    The previous implementation projected each state onto a random float and
    then called ``np.unique``. Besides requiring an O(n log n) sort, that could
    merge distinct states after a projection collision and changed NumPy's
    global random state. Integer count states are now encoded exactly with a
    mixed-radix key and counted in O(n) time when the key space is compact.
    """

    size_factor = np.asarray(size_factor)
    if size_factor.ndim != 1 or size_factor.shape[0] != expr.shape[0]:
        raise ValueError("size_factor must have one finite, positive value per cell")
    if np.any(~np.isfinite(size_factor)) or np.any(size_factor <= 0):
        raise ValueError("size_factor must have one finite, positive value per cell")

    dense_expr = np.asarray(expr.toarray())
    rounded_expr = np.rint(dense_expr)
    integer_counts = (
        np.all(np.isfinite(dense_expr))
        and np.all(dense_expr >= 0)
        and np.array_equal(dense_expr, rounded_expr)
        and (dense_expr.size == 0 or dense_expr.max() <= np.iinfo(np.int64).max)
    )

    if integer_counts:
        count_expr = rounded_expr.astype(np.int64, copy=False)
        sf_code, sf_values = pd.factorize(size_factor, sort=False)
        radices = [int(count_expr[:, idx].max()) + 1 for idx in range(count_expr.shape[1])]

        state_space = len(sf_values)
        for radix in radices:
            state_space *= radix

        if state_space <= np.iinfo(np.int64).max:
            code = sf_code.astype(np.int64, copy=True)
            for idx, radix in enumerate(radices):
                code *= radix
                code += count_expr[:, idx]

            if state_space <= _MAX_DENSE_STATE_BINS:
                all_counts = np.bincount(code)
                unique_code = np.flatnonzero(all_counts)
                counts = all_counts[unique_code]
            else:
                unique_code, counts = np.unique(code, return_counts=True)

            decoded = unique_code.copy()
            unique_expr = np.empty((unique_code.size, count_expr.shape[1]), dtype=count_expr.dtype)
            for idx in range(count_expr.shape[1] - 1, -1, -1):
                unique_expr[:, idx] = decoded % radices[idx]
                decoded //= radices[idx]
            unique_sf = np.asarray(sf_values)[decoded]

            inv_sf = (1 / unique_sf).reshape(-1, 1)
            return inv_sf, inv_sf**2, unique_expr, counts

    # Counts should normally take the fast path. Keep an exact fallback for
    # custom estimators that supply non-integer expression values.
    fields = [("size_factor", size_factor.dtype)]
    fields.extend((f"expr_{idx}", dense_expr.dtype) for idx in range(dense_expr.shape[1]))
    states = np.empty(expr.shape[0], dtype=fields)
    states["size_factor"] = size_factor
    for idx in range(dense_expr.shape[1]):
        states[f"expr_{idx}"] = dense_expr[:, idx]

    _, index, counts = np.unique(states, return_index=True, return_counts=True)
    unique_sf = size_factor[index]
    inv_sf = (1 / unique_sf).reshape(-1, 1)
    return inv_sf, inv_sf**2, dense_expr[index], counts


def _get_batch_size(num_categories, num_boot, batch_size):
    """Choose a bootstrap batch size that bounds the multinomial count array."""

    if batch_size is not None:
        if batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        return min(int(batch_size), num_boot)

    bytes_per_bootstrap = max(1, num_categories) * np.dtype(np.int64).itemsize
    memory_limited_size = max(1, _DEFAULT_BOOTSTRAP_BATCH_BYTES // bytes_per_bootstrap)
    return min(num_boot, memory_limited_size)


def _bootstrap_1d(
    data, 
    size_factor,
    q,
    _estimator_1d,
    num_boot=1000,
    return_times=False,
    batch_size=None,
    rng=None,
    **kwargs):
    """
        Perform the bootstrap and CI calculation for mean and variance.
        
        This function performs bootstrap for a single gene. 
        
        This function expects :data: to be a single sparse column vector.
    """
    start_time = time.time()
    
    # Pre-compute size factor
    # Pass the pre-computed values for permutation test
    inv_sf, inv_sf_sq, expr, counts = _unique_expr(data, size_factor)
    count_time = time.time()
        
    # Skip this gene if it has no expression
    if expr.shape[0] <= 1:
        return np.full(num_boot, np.nan), np.full(num_boot, np.nan)
    
    n_obs = data.shape[0]
        
    # Keep the historical stream for direct internal calls. Public hypothesis
    # tests provide a per-gene generator that advances across biological groups.
    gen = np.random.Generator(np.random.PCG64(5)) if rng is None else rng
    batch_size = _get_batch_size(counts.size, num_boot, batch_size)
    mean = np.empty(num_boot)
    var = np.empty(num_boot)

    for start in range(0, num_boot, batch_size):
        stop = min(start + batch_size, num_boot)
        gene_rvs = gen.multinomial(
            data.shape[0], counts / counts.sum(), size=stop - start).T

        batch_mean, batch_var = _estimator_1d(
            data=(expr, gene_rvs),
            n_obs=n_obs,
            q=q,
            size_factor=(inv_sf, inv_sf_sq))
        mean[start:stop] = batch_mean
        var[start:stop] = batch_var
    boot_time = time.time()
    
    if return_times:
        return start_time, count_time, boot_time

    return mean, var


def _bootstrap_2d(
    data, 
    size_factor,
    q,
    _estimator_1d,
    _estimator_cov,
    num_boot=1000,
    precomputed=None,
    batch_size=None,
    rng=None):
    """
        Perform the bootstrap and CI calculation for covariance and correlation.
    """
    Nc = data.shape[0]

    inv_sf, inv_sf_sq, expr, counts = _unique_expr(data, size_factor) if precomputed is None else precomputed
    
    n_obs = Nc
    gen = np.random.Generator(np.random.PCG64(5)) if rng is None else rng
    batch_size = _get_batch_size(counts.size, num_boot, batch_size)
    cov = np.empty(num_boot)
    var_1 = np.empty(num_boot)
    var_2 = np.empty(num_boot)

    for start in range(0, num_boot, batch_size):
        stop = min(start + batch_size, num_boot)
        gene_rvs = gen.multinomial(
            data.shape[0], counts / counts.sum(), size=stop - start).T

        cov[start:stop] = _estimator_cov(
            data=(expr[:, 0:1], expr[:, 1:2], gene_rvs),
            n_obs=n_obs,
            q=q,
            size_factor=(inv_sf, inv_sf_sq))
        _, var_1[start:stop] = _estimator_1d(
            data=(expr[:, 0:1], gene_rvs),
            n_obs=n_obs,
            q=q,
            size_factor=(inv_sf, inv_sf_sq))
        _, var_2[start:stop] = _estimator_1d(
            data=(expr[:, 1:2], gene_rvs),
            n_obs=n_obs,
            q=q,
            size_factor=(inv_sf, inv_sf_sq))

    return cov, var_1, var_2
