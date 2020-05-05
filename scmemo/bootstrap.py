"""
	bootstrap.py
	
	This file contains functions for fast bootstraping.
"""

import scipy.stats as stats
import numpy as np
import pandas as pd

import estimator

def _sparse_bincount(data):
	""" Sparse bincount. """

	if data.shape[1] == 1: # 1D
		if data.data.shape[0] == 0:
			return np.array([0]), np.array([data.shape[0]])
		counts = np.bincount(data.data)
		counts[0] = data.shape[0] - data.data.shape[0]
		expr = np.arange(counts.shape[0])

		return expr[counts != 0], counts[counts != 0]
	
	else: # 2D
		y = data.dot(np.random.random(2))
		_, index, counts = np.unique(y, return_counts=True, return_index=True)
		return data[index].toarray(), counts, y
		

def _precompute_size_factor(expr, size_factor):
	"""
		Precompute the size factor to separate it from the bootstrap. 
		Must return in the increasing order of :expr:
	"""
	
	df = pd.DataFrame()
	df['expr'] = expr
	df['inv_size_factor'] = 1/size_factor
	df['inv_size_factor_sq'] = 1/size_factor**2
	groupby_obj = df.groupby('expr')
	
	inv_sf = groupby_obj['inv_size_factor'].mean().values
	inv_sf_sq = groupby_obj['inv_size_factor_sq'].mean().values
		
	return inv_sf.reshape(-1, 1), inv_sf_sq.reshape(-1, 1)


def _bootstrap_1d(data, size_factor, num_boot=1000, mv_regressor=None):
	"""
		Perform the bootstrap and CI calculation for mean and variance.
	"""
	
	Nc, G = data.shape
	
	mean_se = np.full(G, np.nan)
	var_se = np.full(G, np.nan)
	res_var_se = np.full(G, np.nan)
	
	for gene_idx in range(G):
		
		# Get expr values and counts
		expr, counts = _sparse_bincount(data[:, gene_idx])
		
		# Skip this gene if it has no expression
		if expr.shape[0] == 1:
			continue
		
		# Generate the bootstrap samples
		gene_mult_rvs = stats.multinomial.rvs(n=Nc, p=counts/Nc, size=num_boot).T
		precomputed_size_factor = _precompute_size_factor(
			expr=data[:, gene_idx].toarray().reshape(-1), 
			size_factor=size_factor)
		
		# Estimate mean and variance
		mean, var = estimator._poisson_1d(
			data=(expr.reshape(-1,1), gene_mult_rvs),
			n_obs=Nc,
			size_factor=precomputed_size_factor)
		
		# Estimate residual variance
		if mv_regressor is not None:
			res_var = estimator._residual_variance(mean, var, mv_regressor)
		else:
			res_var = np.array([np.nan])
		
		mean_se[gene_idx], var_se[gene_idx], res_var_se[gene_idx] = np.nanstd(mean), np.nanstd(var), np.nanstd(res_var)
			
	return mean_se, var_se, res_var_se


def _bootstrap_2d(data, size_factor, gene_idxs_1, gene_idxs_2, num_boot=1000):
	"""
		Perform the bootstrap and CI calculation for covariance and correlation.
	"""
	Nc, G = data.shape
	
	cov_se = np.full((gene_idxs_1.shape[0], gene_idxs_2.shape[0]), np.nan)
	corr_se = np.full((gene_idxs_1.shape[0], gene_idxs_2.shape[0]), np.nan)
	
	for gene_idx_1 in gene_idxs_1:
		for gene_idx_2 in gene_idxs_2:
			
			# Skip if these are the same gene
			if gene_idx_1 == gene_idx_2:
				continue
			
			# Get expr values and counts
			expr, counts, code = _sparse_bincount(data[:, [gene_idx_1, gene_idx_2]])
			
			if expr.shape[0] == 1:
				continue
			
			# Generate the bootstrap samples
			gene_mult_rvs = stats.multinomial.rvs(n=Nc, p=counts/Nc, size=num_boot).T
			precomputed_size_factor = _precompute_size_factor(
				expr=code, 
				size_factor=size_factor)
			
			# Estimate the covariance and variance
			cov = estimator._poisson_cov(
				data=(expr[:, [0]], expr[:, [1]], gene_mult_rvs), 
				n_obs=Nc, 
				size_factor=precomputed_size_factor)
			_, var_1 = estimator._poisson_1d(
				data=(expr[:, [0]], gene_mult_rvs),
				n_obs=Nc,
				size_factor=precomputed_size_factor)
			_, var_2 = estimator._poisson_1d(
				data=(expr[:, [1]], gene_mult_rvs),
				n_obs=Nc,
				size_factor=precomputed_size_factor)
			
			# Convert to correlation
			corr = cov / (np.sqrt(var_1)*np.sqrt(var_2))
			corr[~np.isfinite(corr)] = np.nan
									
			# New indices for this gene set
			new_gene_idx_1 = np.where(gene_idxs_1 == gene_idx_1)[0]
			new_gene_idx_2 = np.where(gene_idxs_2 == gene_idx_2)[0]
			
			cov_se[new_gene_idx_1, new_gene_idx_2] = np.nanstd(cov)
			corr_se[new_gene_idx_1, new_gene_idx_2] = np.nanstd(corr)
			
	return cov_se, corr_se
		