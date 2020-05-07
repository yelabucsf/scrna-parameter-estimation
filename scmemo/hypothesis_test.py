"""
	hypothesis_test.py
	
	This file contains code to perform meta-analysis on the estimated parameters and their confidence intervals.
"""

import numpy as np
import scipy.stats as stats

def wlstsq(X, y, weights):
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


def _ht_1d(design_matrix, response, weights):
	"""
		Performs hypothesis testing. 
		
		Here, weights are standard error of the estimates.
	"""
	
	num_rep, G = response[0].shape
	_, num_param = design_matrix.shape
	
	# Create resonse variables and weights
	mean, var = response
	mean_ci, var_ci = weights
	mean_weights, var_weights = 1/mean_ci**2, 1/var_ci**2
	
	# Initialize array to hold information
	mean_beta = np.full((num_param, G), np.nan)
	mean_stde = np.full((num_param, G), np.nan)
	mean_pval = np.full((num_param, G), np.nan)
	var_beta = np.full((num_param, G), np.nan)
	var_stde = np.full((num_param, G), np.nan)
	var_pval = np.full((num_param, G), np.nan)
	
	# Perform hypothesis test for each gene
	for i in range(G):
		
		mean_beta[:, i], mean_stde[:, i], mean_pval[:, i] = wlstsq(design_matrix, mean[:, i], mean_weights[:, i])
		var_beta[:, i], var_stde[:, i], var_pval[:, i] = wlstsq(design_matrix, var[:, i], var_weights[:, i])
	
	# Return the results
	return [[mean_beta, mean_stde, mean_pval], [var_beta, var_stde, var_pval]]
		
		
		