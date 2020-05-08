"""
	bootstrap.py
	
	This file contains functions for fast bootstraping.
"""

import scipy.stats as stats
import numpy as np
import pandas as pd
import string

import estimator


def _create_size_factor_df(size_factor):
	"""
		Contains the pre-computed size factors for individual cells.
	"""
		
	df = pd.DataFrame(
		data=np.vstack([
			size_factor, 
			1/size_factor, 
			1/size_factor**2]).T,
		columns=['size_factor', 'inv_size_factor', 'inv_size_factor_sq'])
	df['expr'] = 1.0
	df['bin_cutoff'] = 1.0
	df['bin'] = True
	
	return df


def _create_bins(x, num_bins):
	""" Small helper function to create bins dynamically. """
	
# 	bin_edges = np.quantile(x, np.linspace(0, 1, num_bins))

	if x.shape[0] < 20:
		return 1
	_, bin_edges = np.histogram(x, bins=num_bins)
	
	return np.digitize(x, np.unique(bin_edges))

	return pd.cut(x, bins=num_bins, labels=list(string.ascii_lowercase)[:num_bins])


def _precompute_size_factor(expr, sf_df, bins):
	"""
		Precompute the size factor to separate it from the bootstrap.
		This function also serves to find and count unique values.
		
		:sf_df: is already a dictionary with memory already allocated to speed up the calculations.
		It should already include the inverse of size factors and inverse of the squared size factors.
	"""
	
	# Get the expression values into the DataFrame
	if expr.ndim > 1:
		sf_df['expr'] = np.random.random()*expr[:, 0]+np.random.random()*expr[:, 1]
		sf_df['expr1'], sf_df['expr2'] = expr[:, 0], expr[:, 1]
	else:
		sf_df['expr'] = expr
	
	# Create bins for size factors
	if bins == 1:
		sf_df['bin'] = 1
	elif bins == 2:
		sf_df['bin_cutoff'] = sf_df.groupby('expr', sort=True)['size_factor'].transform('mean')
		sf_df['bin'] = sf_df['size_factor'] > sf_df['bin_cutoff']
	else:
		sf_df['bin'] = sf_df.groupby('expr', sort=True)['size_factor'].transform(lambda x: _create_bins(x, bins))
	
	groupby_obj = sf_df.groupby(['expr', 'bin'], sort=True)
	precomputed_sf = groupby_obj[['inv_size_factor', 'inv_size_factor_sq']].mean().dropna()
	
	# Get unique expression values
	if expr.ndim > 1:
		unique_expr = groupby_obj[['expr1', 'expr2']].first().dropna().values
	else:
		unique_expr = precomputed_sf.index.get_level_values(0).values.reshape(-1, 1)
	
	# Get unique counts
	counts = groupby_obj.size().values
					
	return (
		precomputed_sf['inv_size_factor'].values.reshape(-1, 1), 
		precomputed_sf['inv_size_factor_sq'].values.reshape(-1, 1),
		unique_expr,
		counts)


def _bootstrap_1d(
	data, 
	sf_df, 
	num_boot=1000, 
	mv_regressor=None, 
	n_umi=1, 
	bins=2, 
	dirichlet_approx=True,
	log=True):
	"""
		Perform the bootstrap and CI calculation for mean and variance.
		
		This function performs bootstrap for a single gene. 
		
		This function expects :data: to be a single sparse column vector.
	"""
	
	Nc = data.shape[0]
			
	# Pre-compute size factor
	inv_sf, inv_sf_sq, expr, counts = _precompute_size_factor(
		expr=data.toarray().reshape(-1), 
		sf_df=sf_df,
		bins=bins)

	# Skip this gene if it has no expression
	if expr.shape[0] <= bins:
		return np.full(num_boot, np.nan), np.full(num_boot, np.nan)
	
	# Generate the bootstrap samples.
	# The first sample is always the original data counts.
	gen = np.random.Generator(np.random.PCG64(42343))
	gene_rvs = np.zeros((counts.shape[0], num_boot + 1))
	if dirichlet_approx:
		gene_rvs[:, 0] = counts/Nc
		gene_rvs[:, 1:] = gen.dirichlet(alpha=counts, size=num_boot).T
		n_obs = 1
	else:
		gene_rvs[:, 0] = counts
		gene_rvs[:, 1:] = gen.multinomial(n=Nc, pvals=counts/Nc, size=num_boot).T
		n_obs = Nc

	# Estimate mean and variance
	mean, var = estimator._poisson_1d(
		data=(expr, gene_rvs),
		n_obs=n_obs,
		size_factor=(inv_sf, inv_sf_sq),
		n_umi=n_umi)

	# Return the mean and the variance
	if log:
		mean[mean <= 0] = np.nan
		if mv_regressor is not None:
			return np.log(mean), estimator._residual_variance(mean, var, mv_regressor)
		else:
			var[var <= 0] = np.nan
			return np.log(mean), np.log(var)
	else:
		return mean, var


def _bootstrap_2d(data, sf_df, num_boot=1000, n_umi=1, bins=2, dirichlet_approx=True):
	"""
		Perform the bootstrap and CI calculation for covariance and correlation.
	"""
	Nc = data.shape[0]

	inv_sf, inv_sf_sq, expr, counts = _precompute_size_factor(
		expr=data.toarray(), 
		sf_df=sf_df,
		bins=bins)
	
	# Generate the bootstrap samples
	gen = np.random.Generator(np.random.PCG64(42343))
	gene_rvs = np.zeros((counts.shape[0], num_boot + 1))
	if dirichlet_approx:
		gene_rvs[:, 0] = counts/Nc
		gene_rvs[:, 1:] = gen.dirichlet(alpha=counts, size=num_boot).T
		n_obs = 1
	else:
		gene_rvs[:, 0] = counts
		gene_rvs[:, 1:] = gen.multinomial(n=Nc, pvals=counts/Nc, size=num_boot).T
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
			
	return cov, corr, var_1, var_2
		