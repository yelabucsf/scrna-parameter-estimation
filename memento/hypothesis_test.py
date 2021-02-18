"""
	hypothesis_test.py
	
	This file contains code to perform meta-analysis on the estimated parameters and their confidence intervals.
"""

import numpy as np
import scipy.stats as stats
from sklearn.linear_model import LinearRegression

import memento.bootstrap as bootstrap
import memento.estimator as estimator

def _robust_log(val):
	
	val[np.less_equal(val, 0., where=~np.isnan(val))] = np.nanmean(val)
	
	return np.log(val)


def _fill(val):
	
	condition = np.less_equal(val, 0., where=~np.isnan(val)) | np.isnan(val)
	val[condition] = np.random.choice(val[~condition], condition.sum())
	
	return val

def _fill_corr(val):
	
	condition = np.isnan(val)
	val[condition] = np.random.choice(val[~condition], condition.sum())
	
	return val


def _push(val, cond='neg'):
	
	if cond == 'neg':
		nan_idx = val < 0
	else:
		nan_idx = np.isnan(val)
	
	nan_count = nan_idx.sum()
	val[:(val.shape[0]-nan_count)] = val[~nan_idx]
	val[(val.shape[0]-nan_count):] = np.nan
	
	return val


def _compute_asl(perm_diff):
	""" 
		Use the generalized pareto distribution to model the tail of the permutation distribution. 
	"""
	
	null = perm_diff[1:] - perm_diff[0]#np.mean(perm_diff[1:])
	
	stat = perm_diff[0]#np.mean(perm_diff[1:])
	
	if stat > 0:
		extreme_count = (null > stat).sum() + (null < -stat).sum()
	else:
		extreme_count = (null > -stat).sum() + (null < stat).sum()
		
	return (extreme_count+1) / (null.shape[0]+1)
	
	if extreme_count > 10: # We do not need to use the GDP approximation. 

		return (extreme_count+1) / (null.shape[0]+1)

	else: # We use the GDP approximation

		try:

			perm_dist = np.sort(null)# if perm_mean < 0 else np.sort(-perm_diff) # For fitting the GDP later on
			perm_dist = perm_dist[np.isfinite(perm_dist)]
			N_exec = 300 # Starting value for number of exceendences

			while N_exec > 50:

				tail_data = perm_dist[-N_exec:] if stat > 0 else perm_dist[:N_exec]
				params = stats.genextreme.fit(tail_data)
				_, ks_pval = stats.kstest(tail_data, 'genextreme', args=params)

				if ks_pval > 0.05: # roughly a genpareto distribution
					val = stats.genextreme.sf(stat, *params) if stat > 0 else stats.genextreme.cdf(stat, *params)
					return 2 * (N_exec/perm_dist.shape[0]) * val
				else: # Failed to fit genpareto
					N_exec -= 30
			return (extreme_count+1) / (null.shape[0]+1)

		except: # catch any numerical errors

			# Failed to fit genpareto, return the upper bound
			return (extreme_count+1) / (null.shape[0]+1)
		
		
def _ht_1d(
	true_mean, # list of means
	true_res_var, # list of residual variances
	cells, # list of sparse vectors/matrices
	approx_sf, # list of dense arrays
	design_matrix,
	Nc_list,
	num_boot,
	cov_idx,
	mv_fit, # list of tuples
	q, # list of numbers
	_estimator_1d):
	
	good_idxs = np.zeros(design_matrix.shape[0], dtype=bool)
	
	# the bootstrap arrays
	boot_mean = np.zeros((design_matrix.shape[0], num_boot+1))*np.nan
	boot_var = np.zeros((design_matrix.shape[0], num_boot+1))*np.nan

	for group_idx in range(len(true_mean)):

		# Skip if any of the 1d moments are NaNs
		if np.isnan(true_mean[group_idx]) or \
		   np.isnan(true_res_var[group_idx]) or \
		   true_mean[group_idx] == 0 or \
		   true_res_var[group_idx] < 0:
			continue

		# Fill in the true value
		boot_mean[group_idx, 0], boot_var[group_idx, 0] = np.log(true_mean[group_idx]), np.log(true_res_var[group_idx])

		# Generate the bootstrap values
		mean, var = bootstrap._bootstrap_1d(
			data=cells[group_idx],
			size_factor=approx_sf[group_idx],
			num_boot=num_boot,
			q=q[group_idx],
			_estimator_1d=_estimator_1d)
		
		# Compute the residual variance
		res_var = estimator._residual_variance(mean, var, mv_fit[group_idx])
			
		# This replicate is good
		good_idxs[group_idx] = True
		
		# Minimize invalid values
		boot_mean[group_idx, 1:] = np.log(_fill(mean))#_push_nan(mean)#[:num_boot]
		boot_var[group_idx, 1:] = np.log(_fill(res_var))#_push_nan(res_var)#[:num_boot]
		
	# Skip this gene
	if good_idxs.sum() == 0:
		return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
	
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
	
	if boot_var.shape[1] == 0:
		
		print('skipped')
		
		return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
	
	mean_coef = LinearRegression(fit_intercept=False, n_jobs=1)\
		.fit(design_matrix, boot_mean, Nc_list).coef_[:, cov_idx]
	var_coef = LinearRegression(fit_intercept=False, n_jobs=1)\
		.fit(design_matrix, boot_var, Nc_list).coef_[:, cov_idx]
	
	if boot_var.shape[1] < num_boot*0.5:
		return  mean_coef[0], np.nan, np.nan, var_coef[0], np.nan, np.nan

	mean_asl = _compute_asl(mean_coef)
	var_asl = _compute_asl(var_coef)
	
	mean_se = np.nanstd(mean_coef[1:])
	var_se = np.nanstd(var_coef[1:])
		
	return mean_coef[0], mean_se, mean_asl, var_coef[0], var_se, var_asl


def _ht_2d(
	true_corr, # list of correlations for each group
	cells, # list of Nx2 sparse matrices
	approx_sf,
	design_matrix,
	Nc_list,
	num_boot,
	cov_idx,
	q,
	_estimator_1d,
	_estimator_cov):
		
	good_idxs = np.zeros(design_matrix.shape[0], dtype=bool)
	
	# the bootstrap arrays
	boot_corr = np.zeros((design_matrix.shape[0], num_boot+1))*np.nan

	for group_idx in range(design_matrix.shape[0]):

		# Skip if any of the 2d moments are NaNs
		if np.isnan(true_corr[group_idx]) or (np.abs(true_corr[group_idx]) == 1):
			continue

		# Fill in the true value
		boot_corr[group_idx, 0] = true_corr[group_idx]
		
		# Generate the bootstrap values
		cov, var_1, var_2 = bootstrap._bootstrap_2d(
			data=cells[group_idx],
			size_factor=approx_sf[group_idx],
			num_boot=int(num_boot),
			q=q[group_idx],
			_estimator_1d=_estimator_1d,
			_estimator_cov=_estimator_cov)
		
# 		var_1[var_1 < 0] = np.mean(var_1[var_1 > 0])
# 		var_2[var_2 < 0] = np.mean(var_2[var_2 > 0])
				
		corr = estimator._corr_from_cov(cov, var_1, var_2, boot=True)
			
		# This replicate is good
		boot_corr[group_idx, 1:] = corr#[:num_boot]
		vals = _fill_corr(boot_corr[group_idx, :])
		
		# Skip if all NaNs
		if np.all(np.isnan(vals)):
			continue
		
		good_idxs[group_idx] = True
		boot_corr[group_idx, :] = vals

	# Skip this gene
	if good_idxs.sum() == 0:
		return np.nan, np.nan, np.nan
	
	# Skip if each covariate group is not represented
	if np.unique(design_matrix[good_idxs, cov_idx]).shape[0] == 1:
		return np.nan, np.nan, np.nan
	
	vals = _regress_2d(
			design_matrix=design_matrix[good_idxs, :],
			boot_corr=boot_corr[good_idxs, :],
			Nc_list=Nc_list[good_idxs],
			cov_idx=cov_idx)
	
	return vals


def _regress_2d(design_matrix, boot_corr, Nc_list, cov_idx):
	"""
		Performs hypothesis testing for a single pair of genes for many bootstrap iterations.
	"""
	
	num_boot = boot_corr.shape[1]
	
	boot_corr = boot_corr[:, ~np.any(~np.isfinite(boot_corr), axis=0)]
	
	if boot_corr.shape[1] == 0:
		
		return np.nan, np.nan, np.nan
	
	corr_coef = LinearRegression(fit_intercept=False, n_jobs=1)\
		.fit(design_matrix, boot_corr, Nc_list).coef_[:, cov_idx]
	
	if boot_corr.shape[1] < num_boot*0.7:
		return corr_coef[0], np.nan, np.nan

	corr_asl = _compute_asl(corr_coef)
	
	corr_se = np.nanstd(corr_coef[1:])
	
	return corr_coef[0], corr_se, corr_asl
