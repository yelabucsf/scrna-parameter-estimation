"""
	utils.py

	This file contains functions that support scMeMo.
"""


import pandas as pd
import scipy.stats as stats
import numpy as np
import time
import itertools
import scipy as sp
from statsmodels.stats.multitest import fdrcorrection


def _pair(k1, k2, safe=True):
    """
    Cantor pairing function
    http://en.wikipedia.org/wiki/Pairing_function#Cantor_pairing_function
    """
    z = (0.5 * (k1 + k2) * (k1 + k2 + 1) + k2).astype(int)
    return z


def _depair(z):
    """
    Inverse of Cantor pairing function
    http://en.wikipedia.org/wiki/Pairing_function#Inverting_the_Cantor_pairing_function
    """
    w = np.floor((np.sqrt(8 * z + 1) - 1)/2)
    t = (w**2 + w) / 2
    y = (z - t).astype(int)
    x = (w - y).astype(int)
    return x, y


def _sparse_bincount(col):
	""" Sparse bincount. """
	
	if col.data.shape[0] == 0:
		return np.array([col.shape[0]])
	counts = np.bincount(col.data)
	counts[0] = col.shape[0] - col.data.shape[0]

	return counts


def _sparse_cross_covariance(X, Y):
	""" Return the expectation of the product as well as the cross covariance. """

	prod = (X.T*Y).toarray()/X.shape[0]
	cov = prod - np.outer(X.mean(axis=0).A1, Y.mean(axis=0).A1)
	
	return cov, prod


def _compute_1d_statistics(observed, smooth=True):
	""" Compute some non central moments of the observed data. """

	pseudocount = 1/observed.shape[1] if smooth else 0

	first = (observed.sum(axis=0).A1 + pseudocount)/(observed.shape[0]+pseudocount*observed.shape[1])
	c = observed.copy()
	c.data **= 2
	second = (c.sum(axis=0).A1 + pseudocount)/(observed.shape[0]+pseudocount*observed.shape[1])
	del c

	return first, second


def _estimate_mean(observed_first, q):
	""" Estimate the mean vector. """

	return observed_first/q


def _estimate_variance(observed_first, observed_second, mean_inv_allgenes, q, q_sq):
	""" Estimate the true variance. """

	numer = observed_first - (q_sq/q)*observed_first - observed_second
	denom = -q_sq + q*mean_inv_allgenes - q_sq*mean_inv_allgenes

	return numer/denom - observed_first**2/q**2


def _estimate_covariance(observed_first_1, observed_first_2, observed_prod, mean_inv_allgenes, q, q_sq):
	""" Estimate the true covariance. """

	# Estimate covariances except for the diagonal


	denom = q_sq - (q - q_sq)*mean_inv_allgenes
	cov = observed_prod / denom - observed_first_1.reshape(-1,1)@observed_first_2.reshape(1, -1)/q**2

	return cov


def _compute_mean_inv_numis(observed_allgenes_mean, observed_allgenes_variance, q, q_sq):
    """
        Compute the expected value of inverse of N-1.
    """

    denom = observed_allgenes_mean**3 / q**3

    numer = \
        observed_allgenes_variance/q_sq + \
        observed_allgenes_mean/q_sq + \
        observed_allgenes_mean/q + \
        observed_allgenes_mean**2/q**2

    return numer/denom


def _mean_substitution(mat):
	""" Perform mean substition. Get the percentage of missing values. This will lower the power, but should still be unbiased. """

	to_return = mat.copy()
	col_mean = np.nanmean(mat, axis=0)
	col_mean[np.isnan(col_mean)] = 0
	isnan_mat = np.isnan(mat)
	inds = np.where(isnan_mat)
	perc_nans = np.isnan(mat).sum(axis=0)/mat.shape[0]
	to_return[inds] = np.take(col_mean, inds[1])

	return perc_nans, to_return


def _compute_asl(perm_diff):
	""" 
		Use the generalized pareto distribution to model the tail of the permutation distribution. 
	"""

	extreme_count = (perm_diff > 0).sum()
	extreme_count = min(extreme_count, perm_diff.shape[0] - extreme_count)

	if extreme_count > 2: # We do not need to use the GDP approximation. 

		return 2 * ((extreme_count + 1) / (perm_diff.shape[0] + 1))

	else: # We use the GDP approximation

		perm_mean = perm_diff.mean()
		perm_dist = np.sort(perm_diff) if perm_mean < 0 else np.sort(-perm_diff) # For fitting the GDP later on
		N_exec = 300 # Starting value for number of exceendences

		while N_exec > 50:

			tail_data = perm_dist[-N_exec:]
			params = stats.genextreme.fit(tail_data)
			_, ks_pval = stats.kstest(tail_data, 'genextreme', args=params)

			if ks_pval > 0.05: # roughly a genpareto distribution
				return 2 * (N_exec/perm_diff.shape[0]) * stats.genextreme.sf(1, *params)
			else: # Failed to fit genpareto
				N_exec -= 30

		# Failed to fit genpareto, return the upper bound
		return 2 * ((extreme_count + 1) / (perm_diff.shape[0] + 1))


def _robust_linregress(a, b):
	""" Wrapper for scipy linregress function that ignores non-finite values. """

	condition = (np.isfinite(a) & np.isfinite(b))
	x = a[condition]
	y = b[condition]

	return stats.linregress(x,y)


def _fdrcorrect(pvals):
	"""
		Perform FDR correction with nan's.
	"""

	fdr = np.ones(pvals.shape[0])
	_, fdr[~np.isnan(pvals)] = fdrcorrection(pvals[~np.isnan(pvals)])
	return fdr