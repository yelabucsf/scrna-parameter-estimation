"""
	hypothesis_test.py
	
	This file contains code to perform meta-analysis on the estimated parameters and their confidence intervals.
"""

import numpy as np
import scipy.stats as stats
from sklearn.linear_model import LinearRegression

import bootstrap
import estimator


def _simple_wlstsq(w_X_centered, w_X_centered_sq, y, weights):
	
    y_mean = np.average(y, axis=0, weights=weights)    
    beta = (w_X_centered*(y-y_mean)).sum(axis=0)/(w_X_centered_sq).sum(axis=0)
    
    return beta

def _wlstsq(X, y, weights, cov_idx):
	""" 
		Perform weighted least squares and return the coefficients and the p-values. 
		
		Because the standard errors of the estimates are estimated with the bootstrap, we perform the Z-test, trusting these standard errors
		more than we normally would for OLS or WLS. 
	"""

	inv = np.linalg.inv(X.T.dot(weights.reshape(-1,1)*X))
	beta = inv.dot(X.T*weights).dot(y)
# 	stde = np.sqrt(np.diag(inv))
	
	return beta.reshape(-1)[cov_idx]#, stde


def _compute_asl(perm_diff):
	""" 
		Use the generalized pareto distribution to model the tail of the permutation distribution. 
	"""

# 	extreme_count = (perm_diff > 0).sum()
# 	extreme_count = min(extreme_count, perm_diff.shape[0] - extreme_count)
	
# 	c = (perm_diff < 0).mean()
	
# 	return 2*min(1-c, c )
	
# 	stat = perm_diff[0]
# 	boot_stat =  perm_diff[1:]
# 	boot_stat = boot_stat[np.isfinite(boot_stat)]
	
# 	centered = boot_stat - stat

	c = ((perm_diff[1:] > 0).sum() + 1)/(perm_diff[1:].shape[0]+1)
	
	return 2*min(c, 1-c)
	
	return 2 * ((extreme_count + 1) / (perm_diff.shape[0] + 1))
	
	if extreme_count > 2: # We do not need to use the GDP approximation. 

		return 2 * ((extreme_count + 1) / (perm_diff.shape[0] + 1))

	else: # We use the GDP approximation

		try:

			perm_mean = perm_diff.mean()
			perm_dist = np.sort(perm_diff) if perm_mean < 0 else np.sort(-perm_diff) # For fitting the GDP later on
			perm_dist = perm_dist[np.isfinite(perm_dist)]
			N_exec = 300 # Starting value for number of exceendences

			while N_exec > 50:

				tail_data = perm_dist[-N_exec:]
				params = stats.genextreme.fit(tail_data)
				_, ks_pval = stats.kstest(tail_data, 'genextreme', args=params)

				if ks_pval > 0.05: # roughly a genpareto distribution
					return 2 * (N_exec/perm_diff.shape[0]) * stats.genextreme.sf(1, *params)
				else: # Failed to fit genpareto
					N_exec -= 30
			return 2 * ((extreme_count + 1) / (perm_diff.shape[0] + 1))

		except:

			# Failed to fit genpareto, return the upper bound
			return 2 * ((extreme_count + 1) / (perm_diff.shape[0] + 1))

def _ht_1d(
	idx,
	adata_dict, 
	design_matrix,
	Nc_list,
	num_boot,
	cov_idx):
	
	good_idxs = np.zeros(design_matrix.shape[0], dtype=bool)
	
	# the bootstrap arrays
	boot_mean = np.zeros((design_matrix.shape[0], num_boot+1))*np.nan
	boot_var = np.zeros((design_matrix.shape[0], num_boot+1))*np.nan

	for group_idx, group in enumerate(adata_dict['groups']):

		# Skip if any of the 1d moments are NaNs
		if np.isnan(adata_dict['1d_moments'][group][0][idx]) or \
			np.isnan(adata_dict['1d_moments'][group][1][idx]):
			continue

		# Skip if any of the 1d moments are 0s
		if adata_dict['1d_moments'][group][0][idx] == 0 or \
			adata_dict['1d_moments'][group][1][idx] == 0:
			continue

		# This replicate is good
		good_idxs[group_idx] = True

		# Fill in the true value
		boot_mean[group_idx, 0], boot_var[group_idx, 0] = \
			adata_dict['1d_moments'][group][0][idx], adata_dict['1d_moments'][group][2][idx]		

		# Generate the bootstrap values
		data = adata_dict['group_cells'][group][:, idx]
		mean, var = bootstrap._bootstrap_1d(
			data=data,
			size_factor=adata_dict['approx_size_factor'][group],
			num_boot=num_boot,
			n_umi=adata_dict['n_umi'])
		
		boot_mean[group_idx, 1:] = mean
		boot_var[group_idx, 1:] = estimator._residual_variance(mean, var, adata_dict['mv_regressor'][group])
	
# 	print(type(boot_mean), type(boot_var))
# 	return boot_mean, boot_var

	# Skip this gene
	if good_idxs.sum() == 0:
		return np.nan, np.nan, np.nan, np.nan
	
# 	print(boot_var.min(), boot_var.mean(), boot_var.max())
	boot_var[good_idxs,] = np.log(boot_var[good_idxs,]+5)
	
	boot_mean[good_idxs,] = np.log(boot_mean[good_idxs,])

	vals = _regress_1d(
			design_matrix=design_matrix[good_idxs, :],
			boot_mean=boot_mean[good_idxs, :], 
			boot_var=boot_var[good_idxs, :],
			Nc_list=Nc_list[good_idxs],
			cov_idx=cov_idx)
	return vals


def _regress_1d(design_matrix, boot_mean, boot_var, Nc_list, cov_idx):
	"""
		Performs hypothesis testing for a single gene for many bootstrap iterations.
		
		Here, :X_center:, :X_center_Sq:, :boot_var:, :boot_mean: should have the same number of rows
	"""
	
	num_boot = boot_mean.shape[1]
	
	boot_mean = boot_mean[:, ~np.any(~np.isfinite(boot_mean), axis=0)]
	boot_var = boot_var[:, ~np.any(~np.isfinite(boot_var), axis=0)]
	
	if boot_var.shape[1] < num_boot*0.9:
		return np.nan, np.nan, np.nan, np.nan
	
	mean_coef = LinearRegression(fit_intercept=False)\
		.fit(design_matrix, boot_mean, Nc_list).coef_[:, cov_idx]
	var_coef = LinearRegression(fit_intercept=False)\
		.fit(design_matrix, boot_var, Nc_list).coef_[:, cov_idx]

	mean_asl = _compute_asl(mean_coef)
	var_asl = _compute_asl(var_coef)
	
	return mean_coef[0], mean_asl, var_coef[0], var_asl
		