"""
	hypothesis_test.py
	
	This file contains code to perform meta-analysis on the estimated parameters and their confidence intervals.
"""

import numpy as np
import scipy.stats as stats
import scipy.sparse as sparse
from sklearn.linear_model import LinearRegression
import warnings

import memento.bootstrap as bootstrap
import memento.estimator as estimator

def _robust_log(val):
	
	val[np.less_equal(val, 0., where=~np.isnan(val))] = np.nanmean(val)
	
	return np.log(val)


def _fill(val):
	
	condition = np.less_equal(val, 0., where=~np.isnan(val)) | np.isnan(val)
	num_invalid = condition.sum()
	
	if num_invalid == val.shape[0]:
		return None
	
	val[condition] = np.random.choice(val[~condition], num_invalid)
	
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


def _compute_asl(perm_diff, resampling, approx=False):
	""" 
		Use the generalized pareto distribution to model the tail of the permutation distribution. 
	"""
	
	if resampling == 'bootstrap':
	
		null = perm_diff[1:] - perm_diff[0]
	
	else:
		null = perm_diff[1:]
	
	null = null[np.isfinite(null)]
	
	stat = perm_diff[0]
	
	if approx:
		
		null_params = stats.norm.fit(null)
		
		abs_stat = np.abs(stat)
		
		return stats.norm.sf(abs_stat, *null_params) + stats.norm.cdf(-abs_stat, *null_params)

	if stat > 0:
		extreme_count = (null > stat).sum() + (null < -stat).sum()
	else:
		extreme_count = (null > -stat).sum() + (null < stat).sum()

	if extreme_count > 10: # We do not need to use the GDP approximation. 

		return (extreme_count+1) / (null.shape[0]+1)

	else: # We use the GDP approximation
		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			try:

				perm_dist = np.sort(null)# if perm_mean < 0 else np.sort(-perm_diff) # For fitting the GDP later on

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
	covariate,
	treatment,
	Nc_list,
	num_boot,
	mv_fit, # list of tuples
	q, # list of numbers
	_estimator_1d,
	**kwargs):
	
	good_idxs = np.zeros(treatment.shape[0], dtype=bool)
	
	# the resampled arrays
	boot_mean = np.zeros((treatment.shape[0], num_boot+1))*np.nan
	boot_var = np.zeros((treatment.shape[0], num_boot+1))*np.nan

	for group_idx in range(len(true_mean)):

		# Skip if any of the 1d moments are NaNs
		if np.isnan(true_mean[group_idx]) or \
		   np.isnan(true_res_var[group_idx]) or \
		   true_mean[group_idx] == 0 or \
		   true_res_var[group_idx] < 0:
			continue

		# Fill in the true value
		boot_mean[group_idx, 0], boot_var[group_idx, 0] = np.log(true_mean[group_idx]), np.log(true_res_var[group_idx])
		
# 		unique_counts = bootstrap._unique_expr(cells[group_idx], approx_sf[group_idx])
		
		# Generate the bootstrap values (us)
		mean, var = bootstrap._bootstrap_1d(
			data=cells[group_idx],
			size_factor=approx_sf[group_idx],
			num_boot=num_boot,
			q=q[group_idx],
			_estimator_1d=_estimator_1d,
			precomputed=None)
		
		# Compute the residual variance
		res_var = estimator._residual_variance(mean, var, mv_fit[group_idx])
		
		# Minimize invalid values
		filled_mean = _fill(mean)#_push_nan(mean)#[:num_boot]
		filled_var = _fill(res_var)#_push_nan(res_var)#[:num_boot]
		
		# Make sure its a valid replicate
		if filled_mean is None or filled_var is None:
			continue
		
		boot_mean[group_idx, 1:] = np.log(filled_mean)
		boot_var[group_idx, 1:] = np.log(filled_var)
		
		# This replicate is good
		good_idxs[group_idx] = True
		
	# Skip this gene
	if good_idxs.sum() == 0:
		return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
	
# 	return boot_mean #debug
	
	vals = _regress_1d(
			covariate=covariate[good_idxs, :],
			treatment=treatment[good_idxs, :],
			boot_mean=boot_mean[good_idxs, :], 
			boot_var=boot_var[good_idxs, :],
			Nc_list=Nc_list[good_idxs],
			**kwargs)
	return vals


def _cross_coef(A, B, sample_weight):
	
    # Rowwise mean of input arrays & subtract from input arrays themeselves
    A_mA = A - np.average(A, axis=0, weights=sample_weight)
    B_mB = B - np.average(B, axis=0, weights=sample_weight)

    # Sum of squares across rows
    ssA = np.average(A_mA**2, axis=0, weights=sample_weight)

    # Finally get corr coeff
    return A_mA.T.dot(np.diag(sample_weight)).dot(B_mB)/sample_weight.sum() / ssA[:, None]


def _cross_coef_resampled(A, B, sample_weight):
	
    B_mB = B - np.average(B, axis=0, weights=sample_weight)
    A_mA = A - (A*sample_weight[:, :, np.newaxis]).sum(axis=0)/sample_weight.sum(axis=0)[:, np.newaxis]

    # Sum of squares across rows
    ssA = (A_mA**2*sample_weight[:, :, np.newaxis]).sum(axis=0)/sample_weight.sum(axis=0)[:, np.newaxis]

    # temp = np.einsum( 'ij,ijk->kj',  (boot_expr_resampled * weights_resampled), snps_resampled)
    return np.einsum('ijk,ij->jk', A_mA * sample_weight[:, :, np.newaxis], B_mB).T/sample_weight.sum(axis=0) / ssA.T


def _regress_1d(covariate, treatment, boot_mean, boot_var, Nc_list, resample_rep=False,**kwargs):
	"""
		Performs hypothesis testing for a single gene for many bootstrap iterations.
		
		Here, :X_center:, :X_center_Sq:, :boot_var:, :boot_mean: should have the same number of rows
	"""
	
	valid_boostrap_iters = ~np.any(~np.isfinite(boot_mean), axis=0) & ~np.any(~np.isfinite(boot_var), axis=0)
	boot_mean = boot_mean[:, valid_boostrap_iters]
	boot_var = boot_var[:, valid_boostrap_iters]
	
	num_boot = boot_mean.shape[1]-1
	num_rep = boot_mean.shape[0]

	if boot_var.shape[1] == 0:

		print('skipped')

		return [np.zeros(treatment.shape[1])*np.nan]*5

	boot_mean_tilde = boot_mean - LinearRegression(n_jobs=1).fit(covariate,boot_mean, Nc_list).predict(covariate)
	boot_var_tilde = boot_var - LinearRegression(n_jobs=1).fit(covariate,boot_var, Nc_list).predict(covariate)
	treatment_tilde = treatment - LinearRegression(n_jobs=1).fit(covariate, treatment, Nc_list).predict(covariate)
	
	if resample_rep:
		
		replicate_assignment = np.random.choice(num_rep, size=(num_rep, num_boot))
		replicate_assignment[:, 0] = np.arange(num_rep)
		b_iter_assignment = np.random.choice(num_boot, (num_rep, num_boot))+1
		b_iter_assignment[:, 0] = 0
		
		boot_mean_resampled = boot_mean_tilde[(replicate_assignment, b_iter_assignment)]
		boot_var_resampled = boot_var_tilde[(replicate_assignment, b_iter_assignment)]
		treatment_resampled = treatment_tilde[replicate_assignment]
		weights_resampled = Nc_list[replicate_assignment]
		
		mean_coef = _cross_coef_resampled(treatment_resampled, boot_mean_resampled, weights_resampled)
		var_coef = _cross_coef_resampled(treatment_resampled, boot_var_resampled, weights_resampled)	
		
	else:
		
		mean_coef = _cross_coef(treatment_tilde, boot_mean_tilde, Nc_list)
		var_coef = _cross_coef(treatment_tilde, boot_var_tilde, Nc_list)


	mean_asl = np.apply_along_axis(lambda x: _compute_asl(x, **kwargs), 1, mean_coef)
	var_asl = np.apply_along_axis(lambda x: _compute_asl(x, **kwargs), 1, var_coef)

	mean_se = np.nanstd(mean_coef[:, 1:], axis=1)
	var_se = np.nanstd(var_coef[:, 1:], axis=1)

	return mean_coef[:, 0], mean_se, mean_asl, var_coef[:, 0], var_se, var_asl
		
# 	mean_coef = 0
# 	var_coef = 0

# 	strata = np.delete(design_matrix, treatment_idx, axis=1)
# 	uniq_strata = np.unique(strata, axis=0)

# 	for k in range(uniq_strata.shape[0]): # Go through each stratum and get effect size

# 		strata_idx = np.all(strata == uniq_strata[k], axis=1)
# 		boot_mean_k = boot_mean[strata_idx]
# 		boot_var_k = boot_var[strata_idx]
# 		Nc_list_k = Nc_list[strata_idx]


# 		if treatment_idx != [0]:
# 			design_matrix_k = design_matrix[strata_idx][:, treatment_idx + [0]]
# 		else:
# 			design_matrix_k = design_matrix[strata_idx][:, treatment_idx]

# 		mean_coef += LinearRegression(fit_intercept=False, n_jobs=1)\
# 			.fit(design_matrix_k, boot_mean_k, Nc_list_k).coef_[:, 0]*Nc_list_k.sum()
# 		var_coef += LinearRegression(fit_intercept=False, n_jobs=1)\
# 			.fit(design_matrix_k, boot_var_k, Nc_list_k).coef_[:, 0]*Nc_list_k.sum()

# 	mean_coef /= Nc_list.sum()
# 	var_coef /= Nc_list.sum()
	
# 	if boot_var.shape[1] < num_boot*0.5:
# 		return  mean_coef[0], np.nan, np.nan, var_coef[0], np.nan, np.nan

# 	mean_asl = _compute_asl(mean_coef, **kwargs)
# 	var_asl = _compute_asl(var_coef, **kwargs)
	
# 	mean_se = np.nanstd(mean_coef[1:])
# 	var_se = np.nanstd(var_coef[1:])
		
# 	return mean_coef[0], mean_se, mean_asl, var_coef[0], var_se, var_asl


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

		if treatment_idx != [0]:
			design_matrix_k = design_matrix[strata_idx][:, treatment_idx + [0]]
		else:
			design_matrix_k = design_matrix[strata_idx][:, treatment_idx]

		corr_coef += LinearRegression(fit_intercept=False, n_jobs=1)\
			.fit(design_matrix_k, boot_corr_k, Nc_list_k).coef_[:, 0]*Nc_list_k.sum()

	corr_coef /= Nc_list.sum()
	
	if boot_corr.shape[1] < num_boot*0.7:
		return corr_coef[0], np.nan, np.nan

	corr_asl = _compute_asl(corr_coef, **kwargs)
	
	corr_se = np.nanstd(corr_coef[1:])
	
	return corr_coef[0], corr_se, corr_asl
