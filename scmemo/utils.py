"""
	utils.py

	This file contains functions that support scMEMO.
"""


import pandas as pd
import scipy.stats as stats
import scipy.sparse as sparse
import numpy as np
import time
import itertools
import scipy as sp
import logging
from scipy.stats import multivariate_normal
import pickle as pkl
from statsmodels.stats.moment_helpers import cov2corr
from statsmodels.stats.multitest import fdrcorrection


def cross_covariance(X, Y):
	""" Return the expectation of the product as well as the cross covariance. """

	if type(X) != np.ndarray and type(X) != np.matrix:
		X = X.toarray()
		Y = Y.toarray()

	X_mean = X.mean(axis=0)[:, np.newaxis]
	Y_mean = Y.mean(axis=0)[np.newaxis, :]

	cov = np.dot(X.T - X_mean, Y - Y_mean) / (X.shape[0]-1)
	prod = cov + np.dot(X_mean, Y_mean)

	return cov, prod


def get_differential_genes(
	gene_list,
	hypothesis_test_dict, 
	group_1, 
	group_2, 
	which, 
	direction, 
	sig=0.1, 
	num_genes=50):
	"""
		Get genes that are increased in expression in group 2 compared to group 1, sorted in order of significance.
		:which: should be either "mean" or "dispersion"
		:direction: should be either "increase" or "decrease"
		:sig: defines the threshold
		:num_genes: defines the number of genes to be returned. If bigger than the number of significant genes, then return only the significant ones.
	"""

	# Setup keys
	group_key = (group_1, group_2)
	param_key = 'de' if which == 'mean' else 'dv'

	# Find the number of genes to return
	sig_condition = hypothesis_test_dict[group_key][param_key + '_fdr'] < sig
	dir_condition = ((1 if direction == 'increase' else -1)*hypothesis_test_dict[group_key][param_key + '_diff']) > 0
	num_sig_genes = (sig_condition & dir_condition).sum()
	
	# We will order the output by significance, then by effect size. Just turn the FDR of the other half into 1's to remove them from the ordering.
	relevant_fdr = hypothesis_test_dict[group_key][param_key + '_fdr'].copy()
	relevant_fdr[~dir_condition] = 1
	relevant_effect_size = hypothesis_test_dict[group_key][param_key + '_diff'].copy()
	relevant_effect_size[~dir_condition] = 0

	# Get the order of the genes in terms of FDR.
	df = pd.DataFrame()
	df['pval'] = hypothesis_test_dict[group_key][param_key + '_pval'].copy()
	df['fdr'] = relevant_fdr
	df['effect_size'] = np.absolute(relevant_effect_size)
	df['gene'] = gene_list
	df = df.sort_values(by=['fdr', 'effect_size'], ascending=[True, False])
	df['rank'] = np.arange(df.shape[0])+1

	df = df.query('fdr < {}'.format(sig)).iloc[:num_genes, :].copy()

	return df


	return relevant_fdr[order], hypothesis_test_dict[group_key][param_key + '_diff'][order], gene_list[order], order


def compute_asl(perm_diff):
	""" 
		Use the generalized pareto distribution to model the tail of the permutation distribution. 
	"""

	extreme_count = (perm_diff > 0).sum()
	extreme_count = min(extreme_count, perm_diff.shape[0] - extreme_count)

	if extreme_count > 2: # We do not need to use the GDP approximation. 

		return 2 * ((extreme_count + 1) / (perm_diff.shape[0] + 1))

	else: # We use the GDP approximation

		perm_mean = perm_diff.mean()
		perm_dist = np.sort(perm_diff) if perm_mean < 0 else np.sort(-perm_diff) # For fitting the GDP later on
		N_exec = 300 # Starting value for number of exceendences

		while N_exec > 50:

			tail_data = perm_dist[-N_exec:]
			params = stats.genextreme.fit(tail_data)
			_, ks_pval = stats.kstest(tail_data, 'genextreme', args=params)

			if ks_pval > 0.05: # roughly a genpareto distribution
				return 2 * (N_exec/perm_diff.shape[0]) * stats.genextreme.sf(1, *params)
			else: # Failed to fit genpareto
				N_exec -= 30

		# Failed to fit genpareto, return the upper bound
		return 2 * ((extreme_count + 1) / (perm_diff.shape[0] + 1))


def pair(k1, k2, safe=True):
    """
    Cantor pairing function
    http://en.wikipedia.org/wiki/Pairing_function#Cantor_pairing_function
    """
    z = (0.5 * (k1 + k2) * (k1 + k2 + 1) + k2).astype(int)
    return z


def depair(z):
    """
    Inverse of Cantor pairing function
    http://en.wikipedia.org/wiki/Pairing_function#Inverting_the_Cantor_pairing_function
    """
    w = np.floor((np.sqrt(8 * z + 1) - 1)/2)
    t = (w**2 + w) / 2
    y = (z - t).astype(int)
    x = (w - y).astype(int)
    return x, y


def robust_linregress(a, b):
	""" Wrapper for scipy linregress function that ignores non-finite values. """

	condition = (np.isfinite(a) & np.isfinite(b))
	x = a[condition]
	y = b[condition]

	return stats.linregress(x,y)


def fdrcorrect(pvals):
	"""
		Perform FDR correction with nan's.
	"""

	fdr = np.ones(pvals.shape[0])
	_, fdr[~np.isnan(pvals)] = fdrcorrection(pvals[~np.isnan(pvals)])
	return fdr