"""
	bootstrap.py
	
	This file contains functions for fast bootstraping.
"""

import scipy.stats as stats
import numpy as np
import pandas as pd
import string
import time

import estimator

def numpy_fill(arr):
	nan_idxs = np.isnan(arr)
	arr[nan_idxs] = np.nanmedian(arr)

def _unique_expr(expr, size_factor):
	"""
		Precompute the size factor to separate it from the bootstrap.
		This function also serves to find and count unique values.
		
		:expr: is a sparse matrix, either one or two columns
	"""
	
	code = expr.dot(np.random.random(expr.shape[1]))
	
	approx_sf = size_factor
		
	code += np.random.random()*approx_sf
	
	_, index, count = np.unique(code, return_index=True, return_counts=True)
	
	return 1/approx_sf[index].reshape(-1, 1), 1/approx_sf[index].reshape(-1, 1)**2, expr[index].toarray(), count


def _bootstrap_1d(
	data, 
	size_factor,
	num_boot=1000, 
	n_umi=1, 
	return_times=False):
	"""
		Perform the bootstrap and CI calculation for mean and variance.
		
		This function performs bootstrap for a single gene. 
		
		This function expects :data: to be a single sparse column vector.
	"""
	
	Nc = data.shape[0]
				
	# Pre-compute size factor
	inv_sf, inv_sf_sq, expr, counts = _unique_expr(data, size_factor)
		
	# Skip this gene if it has no expression
	if expr.shape[0] <= 1:
		return np.full(num_boot, np.nan), np.full(num_boot, np.nan)
		
	gen = np.random.Generator(np.random.PCG64(5))
	gene_rvs = gen.poisson(counts, size=(num_boot, counts.shape[0])).T
	n_obs = Nc
		
	# Estimate mean and variance
	mean, var = estimator._poisson_1d(
		data=(expr, gene_rvs),
		n_obs=n_obs,
		size_factor=(inv_sf, inv_sf_sq),
		n_umi=n_umi)
	
	if return_times:
		return start_time, count_time, boot_time

	return mean, var


def _bootstrap_2d(
	data, 
	size_factor,
	num_boot=1000, 
	n_umi=1):
	"""
		Perform the bootstrap and CI calculation for covariance and correlation.
	"""
	Nc = data.shape[0]

	inv_sf, inv_sf_sq, expr, counts = _unique_expr(data, size_factor)
	
	# Generate the bootstrap samples
	gen = np.random.Generator(np.random.PCG64(5))
	gene_rvs = gen.poisson(counts, size=(num_boot, counts.shape[0])).T
	n_obs = Nc
	
	# Estimate the covariance and variance
	cov = estimator._poisson_cov(
		data=(expr[:, 0].reshape(-1, 1), expr[:, 1].reshape(-1, 1), gene_rvs), 
		n_obs=n_obs, 
		size_factor=(inv_sf, inv_sf_sq),
		n_umi=n_umi)
	_, var_1 = estimator._poisson_1d(
		data=(expr[:, 0].reshape(-1, 1), gene_rvs),
		n_obs=n_obs,
		size_factor=(inv_sf, inv_sf_sq),
		n_umi=n_umi)
	_, var_2 = estimator._poisson_1d(
		data=(expr[:, 1].reshape(-1, 1), gene_rvs),
		n_obs=n_obs,
		size_factor=(inv_sf, inv_sf_sq),
		n_umi=n_umi)
	
	# Some trickery to allow for the full bootstrap distribution
# 	var_prod = var_1*var_2
# 	sign = np.sign(cov)
# 	corr_sq = cov**2/var_prod
# 	corr = sign*np.sqrt(corr_sq + 1)
		
	# Convert to correlation
	corr = estimator._corr_from_cov(cov, var_1, var_2, boot=True)
		
	return cov, corr, var_1, var_2
		