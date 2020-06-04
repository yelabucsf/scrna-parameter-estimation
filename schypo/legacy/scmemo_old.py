"""
	scmemo.py

	Single Cell Moment Estimator

	This file contains code for implementing the empirical bayes estimator for the Gaussian assumption for true single cell RNA sequencing counts.
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
import sklearn.decomposition as decomp
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt

from utils import *


class SingleCellEstimator(object):
	"""
		SingleCellEstimator is the class for fitting univariate and bivariate single cell data. 

		:beta: Expected value of the capture rate.
	"""


	def __init__(
		self, 
		adata,
		group_label,
		n_umis_column,
		leave_one_out=False,
		num_permute=10000,
		beta=0.1):

		self.anndata = adata.copy()
		self.genes = adata.var.index.values
		self.barcodes = adata.obs.index
		self.beta = beta
		self.group_label = group_label
		self.groups = self.anndata.obs[self.group_label].drop_duplicates().tolist()
		self.group_counts = dict(adata.obs[group_label].value_counts())
		self.leave_one_out = leave_one_out
		if leave_one_out:
			for group in list(self.group_counts.keys()):
				self.group_counts['-' + group] = self.anndata.shape[0] - self.group_counts[group]
		self.n_umis = adata.obs[n_umis_column].values
		self.num_permute = num_permute

		# Initialize parameter containing dictionaries
		self.mean_inv_numis = {}
		self.observed_moments = {}
		self.observed_central_moments = {}
		self.estimated_moments = {}
		self.estimated_central_moments = {}
		self.parameters = {}
		self.parameters_confidence_intervals = {}

		# dictionaries for hypothesis testing
		self.hypothesis_test_result_1d = {}
		self.hypothesis_test_result_2d = {}


	def _select_cells(self, group):
		""" Select the cells. """

		if group == 'all': # All cells
			cell_selector = np.arange(self.anndata.shape[0])
		elif group[0] == '-': # Exclude this group
			cell_selector = (self.anndata.obs[self.group_label] != group[1:]).values
		else: # Include this group
			cell_selector = (self.anndata.obs[self.group_label] == group).values

		return cell_selector


	def _compute_statistics(self, observed, N):
		""" Compute some non central moments of the observed data. """

		# if type(observed) != np.ndarray: # Change this later

		# 	observed = observed.toarray()

		# # Turn the highest value into a 0 for each gene
		# max_idx = observed.argmax(axis=0)
		# observed[max_idx, np.arange(observed.shape[1])] = 0

		if type(observed) != np.ndarray: # Some sparse matrix

			mean = observed.mean(axis=0).A1
			cov = ((observed.T*observed -(sum(observed).T*sum(observed)/N))/(N-1)).todense()

		else: # Numpy ndarray

			mean = observed.mean(axis=0)
			cov = np.cov(observed, rowvar=False)

		prod_expect = cov + mean.reshape(-1,1).dot(mean.reshape(1, -1))

		# Return the first moment, second moment, expectation of the product
		return mean, np.diag(prod_expect), prod_expect


	def _estimate_mean(self, observed_first):
		""" Estimate the mean vector. """

		return observed_first/self.beta


	def _estimate_variance(self, observed_first, observed_second, mean_inv_allgenes):
		""" Estimate the true variance. """

		numer = observed_first - (self.beta_sq/self.beta)*observed_first - observed_second
		denom = -self.beta_sq + self.beta*mean_inv_allgenes - self.beta_sq*mean_inv_allgenes

		return numer/denom - observed_first**2/self.beta**2


	def _estimate_covariance(self, observed_first, observed_second, observed_prod, mean_inv_allgenes):
		""" Estimate the true covariance. """

		# Estimate covariances except for the diagonal
		denom = self.beta_sq - (self.beta - self.beta_sq)*mean_inv_allgenes
		cov = observed_prod / denom - observed_first.reshape(-1,1)@observed_first.reshape(1, -1)/self.beta**2

		# Estimate the variance
		var = self._estimate_variance(observed_first, observed_second, mean_inv_allgenes)

		cov[np.diag_indices_from(cov)] = var
		return var, cov


	def _estimate_residual_variance(self, estimated_mean, estimated_var):
		""" 
			Estimate the residual variance by performing linear regression. 

			Returns the residual variance as well as its logarithm.
		"""

		log_residual_variance = np.log(estimated_var) - (self.mean_var_slope*np.log(estimated_mean) + self.mean_var_inter)

		return np.exp(log_residual_variance), log_residual_variance


	def _compute_mean_inv_numis(self, group):
		"""
			Compute the expected value of inverse of N-1.
		"""

		if group in self.mean_inv_numis:
			return self.mean_inv_numis[group]

		denom = self.observed_moments[group]['allgenes_first']**3 / self.beta**3

		numer = \
			self.observed_central_moments[group]['allgenes_second']/self.beta_sq + \
			self.observed_moments[group]['allgenes_first']/self.beta_sq + \
			self.observed_moments[group]['allgenes_first']/self.beta + \
			self.observed_moments[group]['allgenes_first']**2/self.beta**2

		self.mean_inv_numis[group] = numer/denom

		return self.mean_inv_numis[group]


	def estimate_beta_sq(self, tolerance=5, compute_moments=True):
		""" 
			Estimate the expected value of the square of the capture rate beta. 

			Estimate the relationship between mean and variance. This means that we assume the mean variance relationship to be 
			the same for all cell types. 
		"""

		# Compute observed moments in every group

		if compute_moments:
			groups = self.anndata.obs[self.group_label].drop_duplicates()
			for group in self.groups:

				print('Computing observed moments for:', group)
				self.compute_observed_moments(group)
				#self.compute_observed_moments('-' + group) This is the leave one out part.

		# Combine all observed moments from every group
		x = np.concatenate([self.observed_central_moments[group]['first'] for group in self.groups])
		y = np.concatenate([self.observed_central_moments[group]['second'] for group in self.groups])

		# Filter for finite estimates
		condition = np.isfinite(x) & np.isfinite(y)
		x = x[condition]
		y = y[condition]

		# Estimate the noise level, or CV^2_beta + 1
		noise_level = np.percentile(
			(y/x**2 - 1/x)[y > x], 
			q=tolerance)

		self.all_group_obs_means = x
		self.all_group_obs_var = y
		self.all_group_obs_cv_sq = y/x**2
		self.noise_level = noise_level
		self.beta_sq = (noise_level+1)*self.beta**2


	def plot_cv_mean_curve(self):
		"""
			Plot the observed characteristic relationship between the mean and the coefficient of variation. 

			If an estimate for beta_sq exists, also plot the estimated baseline noise level.
		"""

		plt.figure(figsize=(5.5, 5))
		obs_mean = self.all_group_obs_means
		obs_cv = self.all_group_obs_cv_sq

		plt.scatter(
		    np.log(obs_mean),
		    np.log(obs_cv),
		    s=2
		)

		bound_x = np.arange(
		    np.nanmin(obs_mean),
		    np.nanmax(obs_mean),
		    0.01)
		bound_y = 1/bound_x + self.noise_level

		plt.plot(np.log(bound_x), -np.log(bound_x), color='k', lw=2)
		plt.plot(np.log(bound_x), np.log(bound_y), lw=2, color='r')
		plt.axis('equal');
		plt.legend(['Poisson', 'Poisson + noise', 'genes'])
		plt.title('Observed Mean - CV Relationship');
		plt.xlabel('log( observed mean )')
		plt.ylabel('log( observed CV^2 )')


	def estimate_parameters(self):
		""" Perform parameter estimation. """

		# Compute estimated moments
		for group in self.group_counts:
			self.compute_estimated_moments(group)

		# Combine all estimated moments from every group
		x = np.concatenate([self.estimated_central_moments[group]['first'] for group in self.groups])
		y = np.concatenate([self.estimated_central_moments[group]['second'] for group in self.groups])

		# Filter for finite estimates
		condition = np.isfinite(x) & np.isfinite(y)
		x = x[condition]
		y = y[condition]

		# Estimate the mean-var relationship
		slope, inter, _, _, _ = robust_linregress(np.log(x), np.log(y))
		self.mean_var_slope = slope
		self.mean_var_inter = inter

		# Compute final parameters
		for group in self.group_counts:
			self.compute_params(group)


	def compute_observed_moments(self, group='all'):
		""" Compute the observed statistics. """

		cell_selector = self._select_cells(group)

		observed = self.anndata.X[cell_selector, :].copy()
		N = observed.shape[0]

		first, second, prod = self._compute_statistics(observed, N)
		allgenes_first = self.n_umis[cell_selector].mean()
		allgenes_second = (self.n_umis[cell_selector]**2).mean()

		self.observed_moments[group] = {
			'first':first,
			'second':second,
			'prod':prod,
			'allgenes_first':allgenes_first,
			'allgenes_second':allgenes_second,
		}

		self.observed_central_moments[group] = {
			'first':first,
			'second':second-first**2,
			'prod':prod - first.reshape(-1,1).dot(first.reshape(1, -1)),
			'allgenes_first':allgenes_first,
			'allgenes_second':allgenes_second - allgenes_first**2
		}


	def compute_estimated_moments(self, group='all'):
		""" Use the observed moments to compute the moments of the underlying distribution. """

		mean_inv_numis = self._compute_mean_inv_numis(group)

		estimated_mean = self._estimate_mean(self.observed_moments[group]['first'])

		estimated_var, estimated_cov = self._estimate_covariance(
			self.observed_moments[group]['first'],
			self.observed_moments[group]['second'],
			self.observed_moments[group]['prod'],
			mean_inv_numis)

		self.estimated_central_moments[group] = {
			'first': estimated_mean,
			'second': estimated_var,
			'prod': estimated_cov
		}


	def compute_params(self, group='all'):
		""" 
			Use the estimated moments to compute the parameters of marginal distributions as 
			well as the estimated correlation. 
		"""

		residual_variance, log_residual_variance = self._estimate_residual_variance(
			self.estimated_central_moments[group]['first'],
			self.estimated_central_moments[group]['second'])

		self.parameters[group] = {
			'mean': self.estimated_central_moments[group]['first'],
			'log_mean': np.log(self.estimated_central_moments[group]['first']),
			'residual_var':residual_variance,
			'log_residual_var':log_residual_variance,
			'cov':self.estimated_central_moments[group]['prod'],
			'corr': cov2corr(self.estimated_central_moments[group]['prod'])}


	def compute_confidence_intervals_1d(self, groups=None, groups_to_compare=None, gene_tracker_count=100, timer='off'):
		"""
			Compute 95% confidence intervals around the estimated parameters. 

			Use the Gamma -> Dirichlet framework to speed up the process.

			Uses self.num_permute attribute, same as in hypothesis testing.

			CAVEAT: Uses the same expectation of 1/N as the true value, does not compute this from the permutations. 
			So the result might be slightly off.
		"""
		
		groups_to_iter = self.group_counts.keys() if groups is None else groups
		comparison_groups = groups_to_compare if groups_to_compare else list(itertools.combinations(groups_to_iter, 2))

		mean_inv_numis = {group:self._compute_mean_inv_numis(group) for group in groups_to_iter}

		# Declare placeholders for gene confidence intervals
		parameters = ['mean', 'residual_var', 'log_mean', 'log_residual_var', 'log1p_mean', 'log1p_residual_var']
		ci_dict = {
			group:{param:np.zeros(self.anndata.var.shape[0]) for param in parameters} \
				for group in groups_to_iter}

		# Declare placeholders for group comparison results
		hypothesis_test_parameters = [
			'log_mean_1', 'log_mean_2', 
			'log_residual_var_1', 'log_residual_var_2', 
			'de_diff', 'de_pval', 
			'de_fdr', 'dv_diff', 
			'dv_pval', 'dv_fdr']

		hypothesis_test_dict = {
			(group_1, group_2):{param:np.zeros(self.anndata.var.shape[0]) for param \
				in hypothesis_test_parameters} for group_1, group_2 in comparison_groups}

		# Iterate through each gene and compute a standard error for each gene
		for gene_idx in range(self.anndata.var.shape[0]):
			
			start_time = time.time()

			if gene_tracker_count and gene_idx % gene_tracker_count == 0: 
				print('Computing the {}st/th gene'.format(gene_idx))

			gene_dir_rvs = {}
			gene_counts = {}
			gene_freqs = {}
			for group in groups_to_iter:

				# Grab the values
				cell_selector = self._select_cells(group)
				data = self.anndata.X[cell_selector, :].toarray() if type(self.anndata.X) != np.ndarray else self.anndata.X[cell_selector, :]
				counts = np.bincount(data[:, gene_idx].reshape(-1).astype(int))
				expr_values = np.arange(counts.shape[0])
				expr_values = expr_values[counts != 0]
				counts = counts[counts != 0]
				gene_freqs[group] = counts
				gene_counts[group] = expr_values
			
			counting_time = time.time()
			
			for group in groups_to_iter:
				gene_dir_rvs[group] = stats.dirichlet.rvs(alpha=gene_freqs[group], size=self.num_permute)

			# Construct the repeated values matrix
			values = {group:np.tile(
				gene_counts[group].reshape(1, -1), (self.num_permute, 1)) for group in groups_to_iter}

			# Compute the permuted, observed mean/dispersion
			mean = {group:((gene_dir_rvs[group]) * values[group]).sum(axis=1) for group in groups_to_iter}
			second_moments = {group:((gene_dir_rvs[group]) * values[group]**2).sum(axis=1) for group in groups_to_iter}
			del gene_dir_rvs
			del gene_counts

			# Compute the permuted, estimated moments for both groups
			estimated_means = {group:self._estimate_mean(mean[group]) for group in groups_to_iter}
			estimated_vars = {group:self._estimate_variance(mean[group], second_moments[group], mean_inv_numis[group]) for group in groups_to_iter}
			estimated_residual_vars = {group:self._estimate_residual_variance(estimated_means[group], estimated_vars[group])[0] for group in groups_to_iter}

			# Store the S.E. of the parameter, log(param), and log1p(param)
			for group in groups_to_iter:

				ci_dict[group]['mean'][gene_idx] = np.nanstd(estimated_means[group])
				ci_dict[group]['residual_var'][gene_idx] = np.nanstd(estimated_residual_vars[group])
				ci_dict[group]['log_mean'][gene_idx] = np.nanstd(np.log(estimated_means[group]))
				ci_dict[group]['log_residual_var'][gene_idx] = np.nanstd(np.log(estimated_residual_vars[group]))
				ci_dict[group]['log1p_mean'][gene_idx] = np.nanstd(np.log(estimated_means[group]+1))
				ci_dict[group]['log1p_residual_var'][gene_idx] = np.nanstd(np.log(estimated_residual_vars[group]+1))

			# Perform hypothesis testing
			for group_1, group_2 in comparison_groups:

				# For difference of log means
				boot_log_mean_diff = np.log(estimated_means[group_2]) - np.log(estimated_means[group_1])
				observed_log_mean_diff = self.parameters[group_2]['log_mean'][gene_idx] - self.parameters[group_1]['log_mean'][gene_idx]
				hypothesis_test_dict[(group_1, group_2)]['log_mean_1'][gene_idx]  = self.parameters[group_1]['log_mean'][gene_idx]
				hypothesis_test_dict[(group_1, group_2)]['log_mean_2'][gene_idx]  = self.parameters[group_2]['log_mean'][gene_idx]
				hypothesis_test_dict[(group_1, group_2)]['de_diff'][gene_idx]  = observed_log_mean_diff
				if np.isfinite(observed_log_mean_diff):

					asl = compute_asl(boot_log_mean_diff)

					hypothesis_test_dict[(group_1, group_2)]['de_pval'][gene_idx] = asl
				else:
					hypothesis_test_dict[(group_1, group_2)]['de_pval'][gene_idx] = np.nan

				# For difference of log residual variances
				boot_log_residual_var_diff = np.log(estimated_residual_vars[group_2]) - np.log(estimated_residual_vars[group_1])
				observed_log_residual_var_diff = self.parameters[group_2]['log_residual_var'][gene_idx] - self.parameters[group_1]['log_residual_var'][gene_idx]
				hypothesis_test_dict[(group_1, group_2)]['log_residual_var_1'][gene_idx]  = self.parameters[group_1]['log_residual_var'][gene_idx]
				hypothesis_test_dict[(group_1, group_2)]['log_residual_var_2'][gene_idx]  = self.parameters[group_2]['log_residual_var'][gene_idx]
				hypothesis_test_dict[(group_1, group_2)]['dv_diff'][gene_idx]  = observed_log_residual_var_diff
				if np.isfinite(observed_log_residual_var_diff):
					asl = compute_asl(boot_log_residual_var_diff)
					hypothesis_test_dict[(group_1, group_2)]['dv_pval'][gene_idx] = asl
				else:
					hypothesis_test_dict[(group_1, group_2)]['dv_pval'][gene_idx] = np.nan
			
			storing_time = time.time()

		# Perform FDR correction
		for group_1, group_2 in comparison_groups:

			hypothesis_test_dict[(group_1, group_2)]['de_fdr'] = fdrcorrect(hypothesis_test_dict[(group_1, group_2)]['de_pval'])
			hypothesis_test_dict[(group_1, group_2)]['dv_fdr'] = fdrcorrect(hypothesis_test_dict[(group_1, group_2)]['dv_pval'])

		# Update the attribute dictionaries
		self.parameters_confidence_intervals.update(ci_dict)
		self.hypothesis_test_result_1d.update(hypothesis_test_dict)
		
		if timer == 'on':
			return counting_time - start_time, storing_time - counting_time, sum([values[group].shape[1] for group in groups_to_iter])/len(groups_to_iter)
		

	def compute_confidence_intervals_2d(self, gene_list_1, gene_list_2, groups=None, groups_to_compare=None, gene_tracker_count=100):
		"""
			Compute 95% confidence intervals around the estimated parameters. 

			Use the Gamma -> Dirichlet framework to speed up the process.

			Uses self.num_permute attribute, same as in hypothesis testing.

			CAVEAT: Uses the same expectation of 1/N as the true value, does not compute this from the permutations. 
			So the result might be slightly off.
		"""

		groups_to_iter = self.group_counts.keys() if groups is None else groups
		comparison_groups = groups_to_compare if groups_to_compare else list(itertools.combinations(groups_to_iter, 2))

		mean_inv_numis = {group:self._compute_mean_inv_numis(group) \
			for group in groups_to_iter}

		# Get the gene idx for the two gene lists
		genes_idxs_1 = np.array([np.where(self.anndata.var.index == gene)[0][0] for gene in gene_list_1])
		genes_idxs_2 = np.array([np.where(self.anndata.var.index == gene)[0][0] for gene in gene_list_2])

		# all_pair_counts = {group:set() for group in groups_to_iter}
		pair_counts = {}

		# Declare placeholders for gene confidence intervals
		parameters = ['cov', 'corr']
		ci_dict = {
			group:{param:np.zeros(self.parameters[groups_to_iter[0]]['corr'].shape)*np.nan for param in parameters} \
				for group in groups_to_iter}

		# Declare placeholders for group comparison results
		hypothesis_test_parameters = [
			'dcov_diff', 'dcov_pval', 'dcov_fdr', 'cov_1', 'cov_2',
			'dcorr_diff', 'dcorr_pval', 'dcorr_fdr', 'corr_1', 'corr_2'
			]
		hypothesis_test_dict = {
			(group_1, group_2):{param:np.zeros(self.parameters[group_1]['corr'].shape)*np.nan for param \
				in hypothesis_test_parameters} for group_1, group_2 in comparison_groups}
		for group_1, group_2 in comparison_groups:
			hypothesis_test_dict[(group_1, group_2)]['gene_idx_1'] = genes_idxs_1
			hypothesis_test_dict[(group_1, group_2)]['gene_idx_2'] = genes_idxs_2

		# Iterate through each gene and compute a standard error for each gene
		iter_1 = 0
		iter_2 = 0
		for gene_idx_1 in genes_idxs_1:

			iter_2 = 0
			for gene_idx_2 in genes_idxs_2:

				if gene_idx_2 == gene_idx_1:
					continue

				if (iter_1*genes_idxs_2.shape[0] + iter_2) % gene_tracker_count == 0: 
					print('Computing the {}st/th gene of {}'.format((iter_1*genes_idxs_2.shape[0] + iter_2), genes_idxs_1.shape[0]*genes_idxs_2.shape[0]))

				gene_dir_rvs = {}
				gene_counts = {}
				for group in groups_to_iter:

					# Grab the values
					cell_selector = self._select_cells(group)
					data = self.anndata.X[cell_selector, :].toarray() if type(self.anndata.X) != np.ndarray else self.anndata.X[cell_selector, :]
					cantor_code = pair(data[:, gene_idx_1], data[:, gene_idx_2])

					expr_values, counts = np.unique(cantor_code, return_counts=True)

					pair_counts[group] = expr_values
					gene_dir_rvs[group] = stats.dirichlet.rvs(alpha=counts, size=self.num_permute)

				# # Grab the appropriate Gamma variables given the bincounts of this particular gene
				# gene_gamma_rvs = {group:(gamma_rvs[group][:, np.nonzero(pair_counts[gene_idx_1][gene_idx_2][group][1][:, None] == all_pair_counts_sorted[group])[1]]) \
				# 	for group in groups_to_iter}

				# # Sample dirichlet from the Gamma variables
				# gene_dir_rvs = {group:(gene_gamma_rvs[group]/gene_gamma_rvs[group].sum(axis=1)[:,None]) for group in groups_to_iter}
				# del gene_gamma_rvs

				# Construct the repeated values matrix
				cantor_code = {group:pair_counts[group] for group in groups_to_iter}
				values_1 = {}
				values_2 = {}

				for group in groups_to_iter:
					values_1_raw, values_2_raw = depair(cantor_code[group])
					values_1[group] = np.tile(values_1_raw.reshape(1, -1), (self.num_permute, 1))
					values_2[group] = np.tile(values_2_raw.reshape(1, -1), (self.num_permute, 1))

				# Compute the bootstrapped observed moments
				mean_1 = {group:((gene_dir_rvs[group]) * values_1[group]).sum(axis=1) for group in groups_to_iter}
				second_moments_1 = {group:((gene_dir_rvs[group]) * values_1[group]**2).sum(axis=1) for group in groups_to_iter}
				mean_2 = {group:((gene_dir_rvs[group]) * values_2[group]).sum(axis=1) for group in groups_to_iter}
				second_moments_2 = {group:((gene_dir_rvs[group]) * values_2[group]**2).sum(axis=1) for group in groups_to_iter}
				prod = {group:((gene_dir_rvs[group]) * values_1[group] * values_2[group]).sum(axis=1) for group in groups_to_iter}
				del gene_dir_rvs

				# Compute the permuted, estimated moments for both groups
				estimated_means_1 = {group:self._estimate_mean(mean_1[group]) for group in groups_to_iter}
				estimated_vars_1 = {group:self._estimate_variance(mean_1[group], second_moments_1[group], mean_inv_numis[group]) for group in groups_to_iter}
				estimated_means_2 = {group:self._estimate_mean(mean_2[group]) for group in groups_to_iter}
				estimated_vars_2 = {group:self._estimate_variance(mean_2[group], second_moments_2[group], mean_inv_numis[group]) for group in groups_to_iter}

				# Compute estimated correlations
				estimated_corrs = {}
				estimated_covs = {}
				for group in groups_to_iter:
					denom = self.beta_sq - (self.beta - self.beta_sq)*mean_inv_numis[group]
					cov = prod[group] / denom - (mean_1[group] * mean_2[group])/self.beta**2
					estimated_covs[group] = cov
					estimated_corrs[group] = cov / np.sqrt(estimated_vars_1[group]*estimated_vars_2[group]) 

				# Store the S.E. of the correlation
				for group in groups_to_iter:
					ci_dict[group]['cov'][gene_idx_1, gene_idx_2] = np.nanstd(estimated_covs[group])
					ci_dict[group]['corr'][gene_idx_1, gene_idx_2] = np.nanstd(estimated_corrs[group])

				# Perform hypothesis testing
				for group_1, group_2 in comparison_groups:

					# For difference of covariances
					boot_cov_diff = estimated_covs[group_2] - estimated_covs[group_1]
					observed_cov_diff = self.parameters[group_2]['cov'][gene_idx_1, gene_idx_2] - self.parameters[group_1]['cov'][gene_idx_1, gene_idx_2]
					hypothesis_test_dict[(group_1, group_2)]['cov_1'][gene_idx_1, gene_idx_2]  = self.parameters[group_1]['cov'][gene_idx_1, gene_idx_2]
					hypothesis_test_dict[(group_1, group_2)]['cov_2'][gene_idx_1, gene_idx_2]  = self.parameters[group_2]['cov'][gene_idx_1, gene_idx_2]
					hypothesis_test_dict[(group_1, group_2)]['dcov_diff'][gene_idx_1, gene_idx_2]  = observed_cov_diff

					# For difference of correlations
					boot_corr_diff = estimated_corrs[group_2] - estimated_corrs[group_1]
					observed_corr_diff = self.parameters[group_2]['corr'][gene_idx_1, gene_idx_2] - self.parameters[group_1]['corr'][gene_idx_1, gene_idx_2]
					hypothesis_test_dict[(group_1, group_2)]['corr_1'][gene_idx_1, gene_idx_2]  = self.parameters[group_1]['corr'][gene_idx_1, gene_idx_2]
					hypothesis_test_dict[(group_1, group_2)]['corr_2'][gene_idx_1, gene_idx_2]  = self.parameters[group_2]['corr'][gene_idx_1, gene_idx_2]
					hypothesis_test_dict[(group_1, group_2)]['dcorr_diff'][gene_idx_1, gene_idx_2]  = observed_corr_diff

					if np.isfinite(observed_corr_diff):
						hypothesis_test_dict[(group_1, group_2)]['dcorr_pval'][gene_idx_1, gene_idx_2] = compute_asl(boot_corr_diff)
					if np.isfinite(observed_cov_diff):
						hypothesis_test_dict[(group_1, group_2)]['dcov_pval'][gene_idx_1, gene_idx_2] = compute_asl(boot_cov_diff)

				iter_2 += 1
			iter_1 += 1

		# Perform FDR correction
		for group_1, group_2 in comparison_groups:
			hypothesis_test_dict[(group_1, group_2)]['cov_1'] = hypothesis_test_dict[(group_1, group_2)]['cov_1'][genes_idxs_1, :][:, genes_idxs_2]
			hypothesis_test_dict[(group_1, group_2)]['cov_2'] = hypothesis_test_dict[(group_1, group_2)]['cov_2'][genes_idxs_1, :][:, genes_idxs_2]
			hypothesis_test_dict[(group_1, group_2)]['dcov_diff'] = hypothesis_test_dict[(group_1, group_2)]['dcov_diff'][genes_idxs_1, :][:, genes_idxs_2]
			hypothesis_test_dict[(group_1, group_2)]['dcov_fdr'] = fdrcorrect(hypothesis_test_dict[(group_1, group_2)]['dcov_pval'][genes_idxs_1, :][:, genes_idxs_2].reshape(-1))\
				.reshape(genes_idxs_1.shape[0], genes_idxs_2.shape[0])
			hypothesis_test_dict[(group_1, group_2)]['dcov_pval'] = hypothesis_test_dict[(group_1, group_2)]['dcov_pval'][genes_idxs_1, :][:, genes_idxs_2]

			hypothesis_test_dict[(group_1, group_2)]['corr_1'] = hypothesis_test_dict[(group_1, group_2)]['corr_1'][genes_idxs_1, :][:, genes_idxs_2]
			hypothesis_test_dict[(group_1, group_2)]['corr_2'] = hypothesis_test_dict[(group_1, group_2)]['corr_2'][genes_idxs_1, :][:, genes_idxs_2]
			hypothesis_test_dict[(group_1, group_2)]['dcorr_diff'] = hypothesis_test_dict[(group_1, group_2)]['dcorr_diff'][genes_idxs_1, :][:, genes_idxs_2]
			hypothesis_test_dict[(group_1, group_2)]['dcorr_fdr'] = fdrcorrect(hypothesis_test_dict[(group_1, group_2)]['dcorr_pval'][genes_idxs_1, :][:, genes_idxs_2].reshape(-1))\
				.reshape(genes_idxs_1.shape[0], genes_idxs_2.shape[0])
			hypothesis_test_dict[(group_1, group_2)]['dcorr_pval'] = hypothesis_test_dict[(group_1, group_2)]['dcorr_pval'][genes_idxs_1, :][:, genes_idxs_2]

		# Update the attribute dictionaries
		for group in groups_to_iter:
			if group in self.parameters_confidence_intervals:
				self.parameters_confidence_intervals[group].update(ci_dict[group])
			else:
				self.parameters_confidence_intervals[group] = ci_dict[group]
		self.hypothesis_test_result_2d.update(hypothesis_test_dict)
