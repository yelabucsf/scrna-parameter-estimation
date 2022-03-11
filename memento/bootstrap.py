"""
	bootstrap.py
	
	This file contains functions for fast bootstraping.
"""

import scipy.stats as stats
import numpy as np
import pandas as pd
import string
import time
import statsmodels.api as sm
import warnings


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
		Precompute the size factor to separate it from the bootstrap.
		This function also serves to find and count unique values.
		
		:expr: is a sparse matrix, either one or two columns
		
		FIXIT: we don't need to sorting functionality of np.unique. try to do this faster.
	"""
	
	# Remove outliers
# 	with warnings.catch_warnings():
# 		warnings.simplefilter("ignore")
# 		params = sm.NegativeBinomial(expr.todense().A1, np.ones((expr.shape[0], 1))).fit(disp=0).params
# 	mu_hat = np.exp(params[0])
# 	alpha_hat = params[1]
# 	probs = stats.nbinom.pmf(expr.todense().A1, *_convert_params(mu_hat, alpha_hat))
# 	inliers = probs > 0.1/expr.shape[0]
	
# 	expr = expr#[inliers]
# 	size_factor = size_factor#[inliers]
	
	code = expr.dot(np.random.random(expr.shape[1]))
	approx_sf = size_factor
		
	code += np.random.random()*approx_sf
	
	_, index, count = np.unique(code, return_index=True, return_counts=True)
    
	expr_to_return = expr[index].toarray()
	
	return 1/approx_sf[index].reshape(-1, 1), 1/approx_sf[index].reshape(-1, 1)**2, expr_to_return, count


def _bootstrap_1d(
	data, 
	size_factor,
	q,
	_estimator_1d,
	num_boot=1000,
	return_times=False,
	precomputed=None):
	"""
		Perform the bootstrap and CI calculation for mean and variance.
		
		This function performs bootstrap for a single gene. 
		
		This function expects :data: to be a single sparse column vector.
	"""
	start_time = time.time()
	
	# Pre-compute size factor
	# Pass the pre-computed values for permutation test
	inv_sf, inv_sf_sq, expr, counts = _unique_expr(data, size_factor) if precomputed is None else precomputed
	count_time = time.time()
		
	# Skip this gene if it has no expression
	if expr.shape[0] <= 1:
		return np.full(num_boot, np.nan), np.full(num_boot, np.nan)
	
	n_obs = data.shape[0]
		
	gen = np.random.Generator(np.random.PCG64(5))
	gene_rvs = gen.multinomial(data.shape[0], counts/counts.sum(), size=num_boot).T
		
	# Estimate mean and variance
	mean, var = _estimator_1d(
		data=(expr, gene_rvs),
		n_obs=n_obs,
		q=q,
		size_factor=(inv_sf, inv_sf_sq))
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
	precomputed=None):
	"""
		Perform the bootstrap and CI calculation for covariance and correlation.
	"""
	Nc = data.shape[0]

	inv_sf, inv_sf_sq, expr, counts = _unique_expr(data, size_factor) if precomputed is None else precomputed
	
	# Generate the bootstrap samples
	gen = np.random.Generator(np.random.PCG64(5))
# 	gene_rvs = gen.poisson(counts, size=(num_boot, counts.shape[0])).T
	gene_rvs = gen.multinomial(data.shape[0], counts/counts.sum(), size=num_boot).T
	n_obs = Nc
	
	# Estimate the covariance and variance
	cov = _estimator_cov(
		data=(expr[:, 0].reshape(-1, 1), expr[:, 1].reshape(-1, 1), gene_rvs), 
		n_obs=n_obs, 
		q=q,
		size_factor=(inv_sf, inv_sf_sq))
	_, var_1 = _estimator_1d(
		data=(expr[:, 0].reshape(-1, 1), gene_rvs),
		n_obs=n_obs,
		q=q,
		size_factor=(inv_sf, inv_sf_sq))
	_, var_2 = _estimator_1d(
		data=(expr[:, 1].reshape(-1, 1), gene_rvs),
		n_obs=n_obs,
		q=q,
		size_factor=(inv_sf, inv_sf_sq))
			
	return cov, var_1, var_2
		