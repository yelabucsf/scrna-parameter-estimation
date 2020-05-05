"""
	bootstrap.py
	
	This file contains functions for fast bootstraping.
"""

import scipy.stats as stats
import numpy as np
import pandas as pd
import string

import estimator


def _sparse_bincount(data, size_factor):
	""" Sparse bincount. """
	
	y = data.dot(np.random.random(data.shape[1])) + np.random.random()*size_factor
	_, index, counts = np.unique(y, return_counts=True, return_index=True)
	
	return index, counts
		

def _precompute_size_factor(expr, size_factor, bins):
	"""
		Precompute the size factor to separate it from the bootstrap. 
		Must return in the increasing order of :expr:
	"""
	
	df = pd.DataFrame()
	df['size_factor'] = size_factor
	df['expr'] = expr
	df['bin'] = df.groupby('expr')['size_factor'].transform(lambda x: pd.cut(x, bins=bins, labels=list(string.ascii_lowercase[:bins])))
	df['inv_size_factor'] = df.groupby(['expr', 'bin'])['size_factor'].transform(lambda x: (1/x).mean())
	df['inv_size_factor_sq'] = df.groupby(['expr', 'bin'])['size_factor'].transform(lambda x: (1/x**2).mean())
		
	return df['inv_size_factor'].values, df['inv_size_factor_sq'].values


def _bootstrap_1d(data, size_factor, num_boot=1000, mv_regressor=None, n_umi=1, bins=5):
	"""
		Perform the bootstrap and CI calculation for mean and variance.
	"""
	
	Nc, G = data.shape
	
	mean_se = np.full(G, np.nan)
	var_se = np.full(G, np.nan)
	res_var_se = np.full(G, np.nan)
	
	for gene_idx in range(G):
		
		# Pre-compute size factor
		precomputed_size_factor = _precompute_size_factor(
			expr=data[:, gene_idx].toarray().reshape(-1), 
			size_factor=size_factor,
			bins=bins)
		
		# Get expr values and counts
		index, counts = _sparse_bincount(data[:, gene_idx], precomputed_size_factor[0])
		expr = data[index, gene_idx].toarray()
		precomputed_size_factor = (precomputed_size_factor[0][index].reshape(-1, 1), precomputed_size_factor[1][index].reshape(-1, 1))
		
		# Skip this gene if it has no expression
		if expr.shape[0] == 1:
			continue
		
		# Generate the bootstrap samples
		gene_mult_rvs = stats.multinomial.rvs(n=Nc, p=counts/Nc, size=num_boot).T
		
		# Estimate mean and variance
		mean, var = estimator._poisson_1d(
			data=(expr, gene_mult_rvs),
			n_obs=Nc,
			size_factor=precomputed_size_factor,
			n_umi=n_umi)
		
		# Estimate residual variance
		if mv_regressor is not None:
			res_var = estimator._residual_variance(mean, var, mv_regressor)
		else:
			res_var = np.array([np.nan])
			
		return mean, var, res_var
			
		if np.isfinite(mean).sum() != 0:
			mean_se[gene_idx] = np.nanstd(mean)
		if np.isfinite(var).sum() != 0:
			var_se[gene_idx] = np.nanstd(var)
		if np.isfinite(res_var).sum() != 0:
			res_var_se[gene_idx] = np.nanstd(res_var)
			
	return mean_se, var_se, res_var_se


def _bootstrap_2d(data, size_factor, gene_idxs_1, gene_idxs_2, num_boot=1000, n_umi=1):
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
				size_factor=size_factor,
				n_umi=n_umi)
			
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
			corr = estimator._corr_from_cov(cov, var_1, var_2, boot=True)
									
			# New indices for this gene set
			new_gene_idx_1 = np.where(gene_idxs_1 == gene_idx_1)[0]
			new_gene_idx_2 = np.where(gene_idxs_2 == gene_idx_2)[0]
			
			cov_se[new_gene_idx_1, new_gene_idx_2] = np.nanstd(cov)
			corr_se[new_gene_idx_1, new_gene_idx_2] = np.nanstd(corr)
			
	return cov_se, corr_se
		