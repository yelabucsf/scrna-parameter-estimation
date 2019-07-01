"""
	simplesc.py
	This file contains code for implementing the empirical bayes estimator for the Gaussian assumption for true single cell RNA sequencing counts.
"""


import pandas as pd
import scipy.stats as stats
import numpy as np
import itertools
import scipy as sp
import time
import logging
from scipy.stats import multivariate_normal
import pickle as pkl
from statsmodels.stats.moment_helpers import cov2corr

class SingleCellEstimator(object):
	"""
		SingleCellEstimator is the class for fitting univariate and bivariate single cell data. 
	"""


	def __init__(
		self, 
		adata,
		group_label,
		n_umis_column,
		p=0.1):

		self.anndata = adata.copy()
		self.genes = adata.var.index
		self.barcodes = adata.obs.index
		self.p = p
		self.group_label = group_label
		self.group_counts = dict(adata.obs[group_label].value_counts())
		for group in list(self.group_counts.keys()):
			self.group_counts['-' + group] = self.anndata.shape[0] - self.group_counts[group]
		self.n_umis = adata.obs[n_umis_column].values

		# Initialize parameter containing dictionaries
		self.observed_moments = {}
		self.estimated_moments = {}
		self.estimated_central_moments = {}
		self.parameters = {}


	def _compute_statistics(self, observed, N):
		""" Compute some non central moments of the observed data. """

		# Turn the highest value into a 0
		observed[observed == observed.max()] = 0

		if type(observed) == sp.sparse.csr_matrix:

			mean = observed.mean(axis=0).A1
			cov = ((observed.T*observed -(sum(observed).T*sum(observed)/N))/(N-1)).todense()

		else:

			mean = observed.mean(axis=0)
			cov = np.cov(observed.T)

		prod_expect = cov + mean.reshape(-1,1).dot(mean.reshape(1, -1))

		# Return the first moment, second noncentral moment, expectation of the product
		return mean, np.diag(prod_expect), prod_expect


	def _select_cells(self, group):
		""" Select the cells. """

		if group == 'all': # All cells
			cell_selector = np.arange(self.anndata.shape[0])
		elif group[0] == '-': # Exclude this group
			cell_selector = (self.anndata.obs[self.group_label] != group[1:])
		else: # Include this group
			cell_selector = (self.anndata.obs[self.group_label] == group)

		return cell_selector.values


	def compute_observed_moments(self, group='all'):
		""" Compute the observed statistics. """

		cell_selector = self._select_cells(group)

		observed = self.anndata.X[cell_selector, :].copy()
		N = observed.shape[0]

		first, second, prod = self._compute_statistics(observed, N)

		self.observed_moments[group] = {
			'first':first,
			'second':second,
			'prod':prod,
			'allgenes_first':self.n_umis[cell_selector].mean(),
			'allgenes_second':(self.n_umis[cell_selector]**2).mean(),
		}


	def compute_estimated_moments(self, group='all'):
		""" Use the observed moments to compute the moments of the underlying distribution. """

		mean_inv_numis = self.p * self.observed_moments[group]['allgenes_second'] / self.observed_moments[group]['allgenes_first']**3

		# Find the inverse of the matrix relating the observed and true moments
		moment_map = np.linalg.inv(np.array([[self.p, 0], [self.p - self.p**2, self.p**2 - self.p*mean_inv_numis]]))

		# Find the true moments and save
		true_moments = moment_map.dot(np.vstack([self.observed_moments[group]['first'], self.observed_moments[group]['second']]))
		true_prod = self.observed_moments[group]['prod'] / (self.p**2 - self.p*(1-self.p)*mean_inv_numis)
		true_prod[np.diag_indices_from(true_prod)] = true_moments[1, :]

		self.estimated_moments[group] = {
			'first': true_moments[0, :],
			'second': true_moments[1, :],
			'prod': true_prod}

		self.estimated_central_moments[group] = {
			'first': self.estimated_moments[group]['first'],
			'second': self.estimated_moments[group]['second'] - self.estimated_moments[group]['first']**2,
			'prod': true_prod - self.estimated_moments[group]['first'].reshape(-1, 1).dot(self.estimated_moments[group]['first'].reshape(1, -1))
		}


	# def _tile_symmetric(self, A):
	# 	""" Tile the vector into a square matrix and turn it into a symmetric matrix. """

	# 	return np.tile(A.reshape(-1, 1), (1, A.shape[0])) + np.tile(A.reshape(-1, 1), (1, A.shape[0])).T


	# def _estimate_lognormal(self, observed_mean, observed_cov):
	# 	""" Estimate parameters assuming an underlying lognormal. """

	# 	# Estimate the mean
	# 	estimated_mean = \
	# 		np.log(observed_mean) - \
	# 		np.log(np.ones(self.anndata.shape[1])*self.p) - \
	# 		(1/2) * np.log(
	# 			np.diag(observed_cov)/observed_mean**2 - \
	# 			(1-self.p) / observed_mean + 1)

	# 	# Estimate the variance
	# 	variance_vector = \
	# 		np.log(
	# 			np.diag(observed_cov)/observed_mean**2 - \
	# 			(1-self.p) / observed_mean + 1)

	# 	# Estimate the residual variance
	# 	nonnan_idx = ~np.isnan(estimated_mean) & ~np.isnan(variance_vector)
	# 	slope, intercept, r, p, _ = stats.linregress(estimated_mean[nonnan_idx], np.sqrt(variance_vector[nonnan_idx]))
	# 	estimated_res_var = np.exp(np.sqrt(var) - (means*slope + intercept))

	# 	# Estimate the covariance and fill in variance
	# 	estimated_sigma = np.log(
	# 		observed_cov / (
	# 			self.p**2 * np.exp(
	# 				self._tile_symmetric(estimated_mean) + \
	# 				(1/2)*self._tile_symmetric(variance_vector))
	# 			) + 1)
	# 	estimated_sigma[np.diag_indices_from(estimated_sigma)] = variance_vector

	# 	return estimated_mean, estimated_sigma, estimated_res_var, slope, intercept, r


	# def compute_params(self, group='all'):
	# 	""" Compute 1d and 2d parameters from the data. """

	# 	# Raw statistics should be pre-computed
	# 	assert group in self.observed_mean and group in self.observed_cov

	# 	# Estimate the mean vector
	# 	self.estimated_mean[group], self.estimated_cov[group], self.estimated_res_var[group], slope, intercept, r = \
	# 		self._estimate_lognormal(
	# 			self.observed_mean[group],
	# 			self.observed_cov[group])


	# def compute_permuted_statistics(self, group='all'):
	# 	"""
	# 		Compute permuted statistics for a specified group.
	# 	"""

	# 	permuted_means = []
	# 	permuted_vars = []
	# 	permuted_covs = []

	# 	for i in range(self.num_permute):

	# 		if group == 'all': # All cells
	# 			np.arange(self.anndata.shape[0])
	# 		elif group[0] == '-': # Exclude this group
	# 			cell_selector = (self.perm_group[i] != group[1:])
	# 		else: # Include this group
	# 			cell_selector = (self.perm_group[i] == group)

	# 		observed = self.anndata.X[cell_selector, :]
	# 		N = observed.shape[0]

	# 		observed_mean, observed_cov = self._compute_statistics(observed, N)
	# 		estimated_mean, estimated_cov = self._estimate_lognormal(observed_mean, observed_cov)

	# 		permuted_means.append(estimated_mean)
	# 		permuted_vars.append(np.diag(estimated_cov))

	# 		cov_vector = estimated_cov.copy()
	# 		cov_vector[np.diag_indices_from(cov_vector)] = np.nan
	# 		cov_vector = np.triu(cov_vector)
	# 		cov_vector = cov_vector.reshape(-1)
	# 		cov_vector = cov_vector[~np.isnan(cov_vector)]

	# 		permuted_covs.append(cov_vector)

	# 	self.permutation_statistics[group] = {
	# 		'mean':np.concatenate(permuted_means),
	# 		'var':np.concatenate(permuted_vars),
	# 		'cov':np.concatenate(permuted_covs)}


	# def differential_expression(self, group_1, group_2, method='perm'):
	# 	"""
	# 		Perform transcriptome wide differential expression analysis between two groups of cells.

	# 		Perform permutation test.
	# 	"""

	# 	N_1 = self.group_counts[group_1]
	# 	N_2 = self.group_counts[group_2]

	# 	mu_1 = self.estimated_mean[group_1]
	# 	mu_2 = self.estimated_mean[group_2]

	# 	var_1 = np.diag(self.estimated_cov[group_1])
	# 	var_2 = np.diag(self.estimated_cov[group_2])

	# 	if method == 'perm':

	# 		# Implements the t-test but with permuted null statistics.

	# 		s_delta_var = (var_1/N_1)+(var_2/N_2)
	# 		t_statistic = (mu_1 - mu_2) / np.sqrt(s_delta_var)

	# 		null_s_delta_var = (self.permutation_statistics[group_1]['var']/N_1) + (self.permutation_statistics[group_2]['var']/N_2)
	# 		null_t_statistic = (self.permutation_statistics[group_1]['mean'] - self.permutation_statistics[group_2]['mean']) / np.sqrt(null_s_delta_var)
			
	# 		median_null = np.nanmedian(null_t_statistic)
	# 		pvals = np.array([((null_t_statistic[~np.isnan(null_t_statistic)] > abs_t).mean() + (null_t_statistic[~np.isnan(null_t_statistic)] < -abs_t).mean()) if not np.isnan(abs_t) else np.nan for abs_t in np.absolute(median_null - t_statistic)])
          
	# 		return t_statistic, null_t_statistic, pvals

	# 	elif method == 'bayes':

	# 		prob = stats.norm.sf(
	# 		    0, 
	# 		    loc=self.estimated_mean[group_1] - self.estimated_mean[group_2],
	# 		    scale=np.sqrt(np.diag(self.estimated_cov[group_1]) + np.diag(self.estimated_cov[group_2])))

	# 		return np.log(prob / (1-prob)) * (self.estimated_mean[group_1] != 0.0) * (self.estimated_mean[group_2] != 0.0)

	# 	else:

	# 		logging.info('This method is not yet implemented.')


	# def differential_variance(self, group_1, group_2, method='perm', safe_estimation=True):
	# 	"""
	# 		Perform transcriptome wide differential variance analysis between two groups of cells.

	# 		Uses statistics from the folded Gaussian distribution.
	# 		https://en.wikipedia.org/wiki/Folded_normal_distribution.

	# 		```safe_estimation``` parameter ensures robust differential variance calculation - otherwise,
	# 		we may see a strong mean-variance anticorrelation. 
	# 	"""

	# 	N_1 = self.group_counts[group_1]
	# 	N_2 = self.group_counts[group_2]

	# 	var_1 = np.diag(self.estimated_cov[group_1])
	# 	var_2 = np.diag(self.estimated_cov[group_2])

	# 	if method == 'perm':

	# 		folded_mean_1 = np.sqrt(var_1 * 2 / np.pi)
	# 		folded_mean_2 = np.sqrt(var_2 * 2 / np.pi)

	# 		folded_var_1 = var_1 - folded_mean_1**2
	# 		folded_var_2 = var_2 - folded_mean_1**2

	# 		t_statistic, _ = stats.ttest_ind_from_stats(
	# 			folded_mean_1, 
	# 			folded_var_1, 
	# 			N_1, 
	# 			folded_mean_2,
	# 			folded_var_2,
	# 			N_2,
	# 			equal_var=False)

	# 		null_folded_mean_1 = np.sqrt(self.permutation_statistics[group_1]['var'] * 2 / np.pi)
	# 		null_folded_mean_2 = np.sqrt(self.permutation_statistics[group_2]['var'] * 2 / np.pi)

	# 		null_folded_var_1 = self.permutation_statistics[group_1]['var'] - self.permutation_statistics[group_1]['mean']**2
	# 		null_folded_var_2 = self.permutation_statistics[group_2]['var'] - self.permutation_statistics[group_2]['mean']**2

	# 		null_t_statistic, _ = stats.ttest_ind_from_stats(
	# 			null_folded_mean_1, 
	# 			null_folded_var_1, 
	# 			N_1, 
	# 			null_folded_mean_2,
	# 			null_folded_var_2,
	# 			N_2,
	# 			equal_var=False)

	# 		null_t_statistic = null_t_statistic[self.permutation_statistics[group_1]['mean'] > 0]

	# 		median_null = np.nanmedian(null_t_statistic)
	# 		pvals = np.array([((null_t_statistic[~np.isnan(null_t_statistic)] > abs_t).mean() + (null_t_statistic[~np.isnan(null_t_statistic)] < -abs_t).mean()) if not np.isnan(abs_t) else np.nan for abs_t in np.absolute(median_null - t_statistic)])

	# 		if safe_estimation:

	# 			# Select the cells
	# 			cell_selector_1 = self._select_cells(group_1)
	# 			cell_selector_2 = self._select_cells(group_2)

	# 			# Compute the cell counts in each group that has > 0 expression
	# 			if type(self.anndata.X) == sp.sparse.csr_matrix:
	# 				nonzero_cell_count_1 = (self.anndata.X[cell_selector_1, :] > 0).sum(axis=0).A1
	# 				nonzero_cell_count_2 = (self.anndata.X[cell_selector_2, :] > 0).sum(axis=0).A1
	# 			else:
	# 				nonzero_cell_count_1 = (self.anndata.X[cell_selector_1, :] > 0).sum(axis=0)
	# 				nonzero_cell_count_2 = (self.anndata.X[cell_selector_2, :] > 0).sum(axis=0)					

	# 			# Get masks for genes with insufficient cells
	# 			insufficient_cells_1 = (nonzero_cell_count_1 < self.p*N_1)
	# 			insufficient_cells_2 = (nonzero_cell_count_2 < self.p*N_2)
	# 			insufficient_cells = insufficient_cells_1 | insufficient_cells_2

	# 			# Insert NaNs where we don't have enough cells
	# 			t_statistic[insufficient_cells] = np.nan
	# 			pvals[insufficient_cells] = np.nan

	# 		return t_statistic, null_t_statistic, pvals


	# 	if method == 'levene':

	# 		folded_mean_1 = np.sqrt(var_1 * 2 / np.pi)
	# 		folded_mean_2 = np.sqrt(var_2 * 2 / np.pi)

	# 		folded_var_1 = var_1 - folded_mean_1**2
	# 		folded_var_2 = var_2 - folded_mean_1**2
			
	# 		return stats.ttest_ind_from_stats(
	# 			folded_mean_1, 
	# 			folded_var_1, 
	# 			N_1, 
	# 			folded_mean_2,
	# 			folded_var_2,
	# 			N_2,
	# 			equal_var=False)

	# 	else:

	# 		logging.info('Please pick an available option!')
	# 		return


	# def differential_correlation(self, group_1, group_2, method='pearson'):
	# 	"""
	# 		Differential correlation using Pearson's method.
	# 	"""

	# 	N_1 = self.group_counts[group_1]
	# 	N_2 = self.group_counts[group_2]

	# 	if method == 'pearson':
	# 		corr_1 = cov2corr(self.estimated_cov[group_1])
	# 		corr_2 = cov2corr(self.estimated_cov[group_2])

	# 		z_1 = (1/2) * (np.log(1+corr_1) - np.log(1-corr_1))
	# 		z_2 = (1/2) * (np.log(1+corr_2) - np.log(1-corr_2))

	# 		dz = (z_1 - z_2) / np.sqrt(np.absolute((1/(N_1-3))**2 - (1/(N_2-3))**2))

	# 		return (z_1 - z_2), 2*stats.norm.sf(dz) * (dz > 0) + 2*stats.norm.cdf(dz) * (dz <= 0)

	# 	else:

	# 		logging.info('Not yet implemented!')
	# 		return


	# def save_model(self, file_path='model.pkl'):
	# 	"""
	# 		Save the parameters estimated by the model.
	# 	"""

	# 	save_dict = {}

	# 	save_dict['observed_mean'] = self.observed_mean.copy()
	# 	save_dict['observed_cov'] = self.observed_cov.copy()
	# 	save_dict['estimated_mean'] = self.estimated_mean.copy()
	# 	save_dict['estimated_cov'] = self.estimated_cov.copy()
	# 	save_dict['group_counts'] = self.group_counts.copy()

	# 	pkl.dump(save_dict, open(file_path, 'wb'))


	# def load_model(self, file_path='model.pkl'):
	# 	"""
	# 		Load the parameters estimated by another model.
	# 	"""

	# 	save_dict = pkl.load(open(file_path, 'rb'))

	# 	self.observed_mean = save_dict['observed_mean'].copy()
	# 	self.observed_cov = save_dict['observed_cov'].copy()
	# 	self.estimated_mean = save_dict['estimated_mean'].copy()
	# 	self.estimated_cov = save_dict['estimated_cov'].copy()
	# 	self.group_counts = save_dict['group_counts']
