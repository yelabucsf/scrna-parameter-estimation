"""
	hypothesis_test.py
	
	This file contains code to perform meta-analysis on the estimated parameters and their confidence intervals.
"""

import numpy as np
import scipy.stats as stats
import scipy.sparse as sparse
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


def _compute_asl(perm_diff, resampling, nonneg, approx=False):
	""" 
		Use the generalized pareto distribution to model the tail of the permutation distribution. 
	"""
	
	if resampling == 'bootstrap':
		null = perm_diff[1:] - perm_diff[0]
	else:
		if nonneg:
			null = perm_diff[1:]
		else:
			null = perm_diff[1:] - perm_diff[1:].mean()
	
	stat = perm_diff[0]
	
	if approx and not nonneg:
		
		null_params = stats.norm.fit(null)
		
		abs_stat = np.abs(stat)
		
		return stats.norm.sf(abs_stat, *null_params) + stats.norm.cdf(-abs_stat, *null_params)
	
	elif approx and nonneg:
		
		null_params = stats.gamma.fit(null)
		
		return stats.gamma.sf(stat, *null_params)
	
	elif nonneg:
		
		extreme_count = (null > stat).sum()
		
		if extreme_count > 10:
			
			return (extreme_count + 1) / (null.shape[0]+1)
		
		else:
			
			try:

				perm_dist = np.sort(null)# if perm_mean < 0 else np.sort(-perm_diff) # For fitting the GDP later on
				perm_dist = perm_dist[np.isfinite(perm_dist)]

				# Right tail
				N_exec = 300
				while N_exec > 50:

					tail_data = perm_dist[-N_exec:]
					params = stats.genextreme.fit(tail_data)
					_, ks_pval = stats.kstest(tail_data, 'genextreme', args=params)

					if ks_pval > 0.05: # roughly a genpareto distribution
						val = stats.genextreme.sf(np.abs(stat), *params)
						right_asl = (N_exec/perm_dist.shape[0]) * val
						return right_asl
					else: # Failed to fit genpareto
						N_exec -= 30					

				return (extreme_count+1) / (null.shape[0]+1)

			except: # catch any numerical errors

				# Failed to fit genpareto, return the upper bound
				return (extreme_count+1) / (null.shape[0]+1)
	else:

		if stat > 0:
			extreme_count = (null > stat).sum() + (null < -stat).sum()
		else:
			extreme_count = (null > -stat).sum() + (null < stat).sum()

		if extreme_count > 10: # We do not need to use the GDP approximation. 

			return (extreme_count+1) / (null.shape[0]+1)

		else: # We use the GDP approximation

			try:

				perm_dist = np.sort(null)# if perm_mean < 0 else np.sort(-perm_diff) # For fitting the GDP later on
				perm_dist = perm_dist[np.isfinite(perm_dist)]

				# Left tail
				N_exec = 300
				left_fit = False
				while N_exec > 50:

					tail_data = perm_dist[:N_exec]
					params = stats.genextreme.fit(tail_data)
					_, ks_pval = stats.kstest(tail_data, 'genextreme', args=params)

					if ks_pval > 0.05: # roughly a genpareto distribution
						val = stats.genextreme.cdf(-np.abs(stat), *params)
						left_asl = (N_exec/perm_dist.shape[0]) * val
						left_fit = True
						break
					else: # Failed to fit genpareto
						N_exec -= 30

				if not left_fit:
					return (extreme_count+1) / (null.shape[0]+1)

				# Right tail
				N_exec = 300
				while N_exec > 50:

					tail_data = perm_dist[-N_exec:]
					params = stats.genextreme.fit(tail_data)
					_, ks_pval = stats.kstest(tail_data, 'genextreme', args=params)

					if ks_pval > 0.05: # roughly a genpareto distribution
						val = stats.genextreme.sf(np.abs(stat), *params)
						right_asl = (N_exec/perm_dist.shape[0]) * val
						return right_asl + left_asl
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
	treatment_idx,
	mv_fit, # list of tuples
	q, # list of numbers
	_estimator_1d,
	resampling,
	**kwargs):
	
	good_idxs = np.zeros(design_matrix.shape[0], dtype=bool)
	
	# the resampled arrays
	boot_mean = np.zeros((design_matrix.shape[0], num_boot+1))*np.nan
	boot_var = np.zeros((design_matrix.shape[0], num_boot+1))*np.nan
	
	# Get strata-specific pooled information
	if resampling == 'permutation':
		
		uniq_strata, strata_indicator = np.unique(np.delete(design_matrix, treatment_idx, axis=1), axis=0, return_inverse=True)
		resampling_info = {}
		
		for k in range(uniq_strata.shape[0]):
			
			strata_idx = np.where(strata_indicator==0)[0]
			data_list = [cells[i] for i in strata_idx]
			sf_list = [approx_sf[i] for i in strata_idx]
		
			resampling_info[k] = bootstrap._unique_expr(sparse.vstack(data_list, format='csc'), np.concatenate(sf_list))

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
			_estimator_1d=_estimator_1d,
			precomputed= (None if resampling == 'bootstrap' else resampling_info[strata_indicator[group_idx]]))
		
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
			treatment_idx=treatment_idx,
			resampling=resampling,
			**kwargs)
	return vals


def _regress_1d(design_matrix, boot_mean, boot_var, Nc_list, treatment_idx, **kwargs):
	"""
		Performs hypothesis testing for a single gene for many bootstrap iterations.
		
		Here, :X_center:, :X_center_Sq:, :boot_var:, :boot_mean: should have the same number of rows
	"""
	
	num_boot = boot_mean.shape[1]
	nonneg = False
	
	boot_mean = boot_mean[:, ~np.any(~np.isfinite(boot_mean), axis=0)]
	boot_var = boot_var[:, ~np.any(~np.isfinite(boot_var), axis=0)]
	
	if boot_var.shape[1] == 0:
		
		print('skipped')
		
		return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
		
	mean_coef = 0
	var_coef = 0

	strata = np.delete(design_matrix, treatment_idx, axis=1)
	uniq_strata = np.unique(strata, axis=0)

	for k in range(uniq_strata.shape[0]): # Go through each stratum and get effect size

		strata_idx = np.all(strata == uniq_strata[k], axis=1)
		boot_mean_k = boot_mean[strata_idx]
		boot_var_k = boot_var[strata_idx]
		Nc_list_k = Nc_list[strata_idx]

		if len(treatment_idx) == 1:

			if treatment_idx != [0]:
				design_matrix_k = design_matrix[strata_idx][:, treatment_idx + [0]]
			else:
				design_matrix_k = design_matrix[strata_idx][:, treatment_idx]

			mean_coef += LinearRegression(fit_intercept=False, n_jobs=1)\
				.fit(design_matrix_k, boot_mean_k, Nc_list_k).coef_[:, 0]*Nc_list_k.sum()
			var_coef += LinearRegression(fit_intercept=False, n_jobs=1)\
				.fit(design_matrix_k, boot_var_k, Nc_list_k).coef_[:, 0]*Nc_list_k.sum()
		else: # Categorical treatment

			nonneg=True

			bm = boot_mean_k# * Nc_list_k.reshape(-1,1)
			bv = boot_var_k #* Nc_list_k.reshape(-1,1)

			mean_coef += (bm.max(axis=0) - bm.min(axis=0))*Nc_list_k.sum()
			var_coef += (bv.max(axis=0) - bv.min(axis=0))*Nc_list_k.sum()

	mean_coef /= Nc_list.sum()
	var_coef /= Nc_list.sum()
	
	if boot_var.shape[1] < num_boot*0.5:
		return  mean_coef[0], np.nan, np.nan, var_coef[0], np.nan, np.nan

	mean_asl = _compute_asl(mean_coef, nonneg=nonneg, **kwargs)
	var_asl = _compute_asl(var_coef, nonneg=nonneg, **kwargs)
	
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
	treatment_idx,
	q,
	_estimator_1d,
	_estimator_cov,
	resampling,
	**kwargs):
	
		
	good_idxs = np.zeros(design_matrix.shape[0], dtype=bool)
	
	# the bootstrap arrays
	boot_corr = np.zeros((design_matrix.shape[0], num_boot+1))*np.nan
	
	# Get strata-specific pooled information
	if resampling == 'permutation':
		
		uniq_strata, strata_indicator = np.unique(np.delete(design_matrix, treatment_idx, axis=1), axis=0, return_inverse=True)
		resampling_info = {}
		
		for k in range(uniq_strata.shape[0]):
			
			strata_idx = np.where(strata_indicator==0)[0]
			data_list = [cells[i] for i in strata_idx]
			sf_list = [approx_sf[i] for i in strata_idx]
		
			resampling_info[k] = bootstrap._unique_expr(sparse.vstack(data_list, format='csc'), np.concatenate(sf_list))

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
			_estimator_cov=_estimator_cov,
			precomputed=(None if resampling == 'bootstrap' else resampling_info[strata_indicator[group_idx]]))
				
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
	
	vals = _regress_2d(
			design_matrix=design_matrix[good_idxs, :],
			boot_corr=boot_corr[good_idxs, :],
			Nc_list=Nc_list[good_idxs],
			treatment_idx=treatment_idx,
			resampling=resampling,
			**kwargs)
	
	return vals


def _regress_2d(design_matrix, boot_corr, Nc_list, treatment_idx, **kwargs):
	"""
		Performs hypothesis testing for a single pair of genes for many bootstrap iterations.
	"""
		
	num_boot = boot_corr.shape[1]
	nonneg = False
	
	boot_corr = boot_corr[:, ~np.any(~np.isfinite(boot_corr), axis=0)]
	
	if boot_corr.shape[1] == 0:
		
		return np.nan, np.nan, np.nan
		
	corr_coef = 0

	strata = np.delete(design_matrix, treatment_idx, axis=1)
	uniq_strata = np.unique(strata, axis=0)

	for k in range(uniq_strata.shape[0]): # Go through each stratum and get effect size

		strata_idx = np.all(strata == uniq_strata[k], axis=1)
		boot_corr_k = boot_corr[strata_idx]
		Nc_list_k = Nc_list[strata_idx]

		if len(treatment_idx) == 1:

			if treatment_idx != [0]:
				design_matrix_k = design_matrix[strata_idx][:, treatment_idx + [0]]
			else:
				design_matrix_k = design_matrix[strata_idx][:, treatment_idx]

			corr_coef += LinearRegression(fit_intercept=False, n_jobs=1)\
				.fit(design_matrix_k, boot_corr_k, Nc_list_k).coef_[:, 0]*Nc_list_k.sum()

		else: # Categorical treatment

			nonneg = True

			bc = boot_corr_k# * Nc_list_k.reshape(-1,1)

			corr_coef += (bc.max(axis=0) - bc.min(axis=0))*Nc_list_k.sum()

	corr_coef /= Nc_list.sum()
	
	if boot_corr.shape[1] < num_boot*0.7:
		return corr_coef[0], np.nan, np.nan

	corr_asl = _compute_asl(corr_coef, nonneg=nonneg, **kwargs)
	
	corr_se = np.nanstd(corr_coef[1:])
	
	return corr_coef[0], corr_se, corr_asl
