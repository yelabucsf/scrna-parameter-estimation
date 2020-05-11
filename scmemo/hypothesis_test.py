"""
	hypothesis_test.py
	
	This file contains code to perform meta-analysis on the estimated parameters and their confidence intervals.
"""

import numpy as np
import scipy.stats as stats
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression

def _wlstsq(X, y, weights):
	""" 
		Perform weighted least squares and return the coefficients and the p-values. 
		
		Because the standard errors of the estimates are estimated with the bootstrap, we perform the Z-test, trusting these standard errors
		more than we normally would for OLS or WLS. 
	"""

	inv = np.linalg.inv(X.T.dot(weights.reshape(-1,1)*X))
	beta = inv.dot(X.T*weights).dot(y)
	stde = np.sqrt(np.diag(inv))/3

	pval = 2*(1-stats.norm.cdf(np.abs(beta), 0, stde))

	return beta, stde, pval


def _compute_asl(perm_diff):
	""" 
		Use the generalized pareto distribution to model the tail of the permutation distribution. 
	"""

	extreme_count = (perm_diff > 0).sum()
	extreme_count = min(extreme_count, perm_diff.shape[0] - extreme_count)
	
# 	return 2 * ((extreme_count + 1) / (perm_diff.shape[0] + 1))
	
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


def _ht_1d(design_matrix, boot_mean, boot_var, Nc_list, cov_idx):
	"""
		Performs hypothesis testing for a single gene for many bootstrap iterations.
		
		Here, :design_matrix:, :boot_mean:, :boot_var: should have the same number of rows
	"""
	
	# Get some constants
	num_rep, num_boot = boot_mean.shape
	_, num_param = design_matrix.shape
	
	# Impute NaNs with the column average
	imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
	boot_mean = imputer.fit_transform(boot_mean)
	boot_var = imputer.fit_transform(boot_var)
	
	# Fit the linear models for all bootstrap iterations
	if design_matrix.shape[0] > 2:
		mean_coef = LinearRegression(fit_intercept=False)\
			.fit(design_matrix, boot_mean, Nc_list).coef_[:, cov_idx]
		var_coef = LinearRegression(fit_intercept=False)\
			.fit(design_matrix, boot_var, Nc_list).coef_[:, cov_idx]
	elif design_matrix.shape[0] == 2:
		ctrl_row = int(design_matrix[0, cov_idx] == 1)
		mean_coef = boot_mean[1-ctrl_row, :] - boot_mean[ctrl_row, :]
		var_coef = boot_var[1-ctrl_row, :] - boot_var[ctrl_row, :]
	else:
		return np.nan, np.nan, np.nan, np.nan
	
	mean_asl = _compute_asl(mean_coef)
	var_asl = _compute_asl(var_coef)
	
	return mean_coef[0], mean_asl, var_coef[0], var_asl
		