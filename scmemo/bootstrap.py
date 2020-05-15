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

def _unique_expr(expr, size_factor, bins=None):
	"""
		Precompute the size factor to separate it from the bootstrap.
		This function also serves to find and count unique values.
		
		:expr: is a sparse matrix, either one or two columns
	"""
	
	code = expr.dot(np.random.random(expr.shape[1]))

# 	if bins is None:
# 		if expr.shape[1] == 1:
# 			num_unique = len(set(expr.data))
# 		else:
# 			num_unique = np.unique(code).shape[0]

# 		bins = 30
	
# 	_, sf_bin_edges = np.histogram(size_factor, bins=bins)
# 	binned_stat = stats.binned_statistic(size_factor, size_factor, bins=bins, statistic='mean')
# 	bin_idx = np.clip(binned_stat[2], a_min=1, a_max=binned_stat[0].shape[0])
# 	approx_sf = binned_stat[0][bin_idx-1]
# 	max_sf = size_factor.max()
# 	approx_sf[size_factor == max_sf] = max_sf
	
	approx_sf = size_factor
	
	start_time = time.time()
	
	code += np.random.random()*approx_sf
	
	_, index, count = np.unique(code, return_index=True, return_counts=True)
	
	return 1/approx_sf[index].reshape(-1, 1), 1/approx_sf[index].reshape(-1, 1)**2, expr[index].toarray(), count, start_time


def _bootstrap_1d(
	data, 
	size_factor,
	true_mean, 
	true_var,
	num_boot=1000, 
	n_umi=1, 
	dirichlet_approx=False,
	bins=None,
	return_times=False):
	"""
		Perform the bootstrap and CI calculation for mean and variance.
		
		This function performs bootstrap for a single gene. 
		
		This function expects :data: to be a single sparse column vector.
	"""
	
	Nc = data.shape[0]
				
	# Pre-compute size factor
	inv_sf, inv_sf_sq, expr, counts, start_time = _unique_expr(data, size_factor, bins=bins)
		
	# Skip this gene if it has no expression
	if expr.shape[0] <= 1:
		return np.full(num_boot, np.nan), np.full(num_boot, np.nan)
	
	count_time = time.time()
	
# 	boot_means, boot_vars = [], []
	
# 	for i in range(num_boot):
		
# 		boot_idxs = np.random.choice(data.shape[0], data.shape[0])
		
# 		res = estimator._poisson_1d(
# 			data=data[boot_idxs],
# 			n_obs=data.shape[0],
# 			size_factor=size_factor[boot_idxs],
# 			n_umi=n_umi)
		
# 		boot_means.append(res[0][0])
# 		boot_vars.append(res[1][0])
	
# 	mean = np.array(boot_means)
# 	var = np.array(boot_vars)
			
	
	# Generate the bootstrap samples.
	# The first sample is always the original data counts.
	gen = np.random.Generator(np.random.PCG64(np.random.randint(10000)))
	if dirichlet_approx:
		gene_rvs = gen.dirichlet(alpha=counts, size=num_boot).T
		n_obs = 1
	else:
# 		gene_rvs = gen.multinomial(n=Nc, pvals=counts/Nc, size=num_boot).T
		gene_rvs = stats.poisson.rvs(counts, size=(num_boot, counts.shape[0])).T
		n_obs = Nc
		
	# Estimate mean and variance
	mean, var = estimator._poisson_1d(
		data=(expr, gene_rvs),
		n_obs=n_obs,
		size_factor=(inv_sf, inv_sf_sq),
		n_umi=n_umi)

	# Bias correction
# 	mean += true_mean - mean.mean()
# 	var += true_var - var.mean()
		
	boot_time = time.time()
	
	if return_times:
		return start_time, count_time, boot_time

	return mean, var


def _bootstrap_2d(
	data, 
	size_factor,
	true_cov,
	true_corr,
	num_boot=1000, 
	n_umi=1, 
	dirichlet_approx=True,
	bins=None):
	"""
		Perform the bootstrap and CI calculation for covariance and correlation.
	"""
	Nc = data.shape[0]

	inv_sf, inv_sf_sq, expr, counts = _unique_expr(data, size_factor, bins=bins)
	
	# Generate the bootstrap samples
	gen = np.random.Generator(np.random.PCG64(42343))
	if dirichlet_approx:
		gene_rvs = gen.dirichlet(alpha=counts, size=num_boot).T
		n_obs = 1
	else:
		gene_rvs = gen.multinomial(n=Nc, pvals=counts/Nc, size=num_boot).T
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
	
	# Convert to correlation
	corr = estimator._corr_from_cov(cov, var_1, var_2, boot=True)

	# Bias correction
	corr += true_corr - corr.mean()
	cov += true_cov - cov.mean()
	
	return cov, corr, var_1, var_2
		