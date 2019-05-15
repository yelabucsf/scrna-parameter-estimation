"""
	simplesc.py
	This file contains code for implementing the empirical bayes estimator for the Gaussian assumption for true single cell RNA sequencing counts.
"""


import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import numpy as np
import itertools
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
		p=0.05,
		group_label='leiden'):

		self.anndata = adata
		self.genes = adata.var.index
		self.barcodes = adata.obs.index
		self.p = p
		self.group_label = group_label
		self.observed_mus = {}
		self.observed_sigmas = {}
		self.estimated_mus = {}
		self.estimated_sigmas = {}
		self.diff_exp = {}



	def compute_observed_statistics(self, group='all'):
		""" Compute the observed statistics. """

		cell_selector = (self.anndata.obs[self.group_label] == group) if group != 'all' else np.arange(self.anndata.shape[0])

		observed = self.anndata.X[cell_selector.values, :]
		N = observed.shape[0]

		self.observed_mus[group] = observed.mean(axis=0).A1
		self.observed_sigmas[group] = ((observed.T*observed -(sum(observed).T*sum(observed)/N))/(N-1)).todense()


	def _tile_symmetric(self, A):
		""" Tile the vector into a square matrix and turn it into a symmetric matrix. """

		return np.tile(A.reshape(-1, 1), (1, A.shape[0])) + np.tile(A.reshape(-1, 1), (1, A.shape[0])).T


	def compute_params(self, group='all'):
		""" Compute 1d and 2d parameters from the data. """

		# Raw statistics should be pre-computed
		assert group in self.observed_mus and group in self.observed_sigmas

		# Estimate the mean vector
		self.estimated_mus[group] = \
			np.log(self.observed_mus[group]) - \
			np.log(np.ones(self.anndata.shape[1])*self.p) - \
			(1/2) * np.log(
				np.diag(self.observed_sigmas[group])/self.observed_mus[group]**2 - \
				(1-self.p) / self.observed_mus[group] + 1)

		# Estimate the variance vector
		variance_vector = \
			np.log(
				np.diag(self.observed_sigmas[group])/self.observed_mus[group]**2 - \
				(1-self.p) / self.observed_mus[group] + 1)

		# Estimate the covariance and fill in variance
		sigma = np.log(
			self.observed_sigmas[group] / (
				self.p**2 * np.exp(
					self._tile_symmetric(self.estimated_mus[group]) + \
					(1/2)*self._tile_symmetric(variance_vector))
				) + 1)
		sigma[np.diag_indices_from(sigma)] = variance_vector
		self.estimated_sigmas[group] = sigma

		# Fill NaN's with 0s. These arise from genes with insufficient/no counts.
		self.estimated_mus[group] = np.nan_to_num(self.estimated_mus[group])
		self.estimated_sigmas[group] = np.nan_to_num(self.estimated_sigmas[group])


	def generate_transcriptome(self, group='all'):
		""" 

			Generate a transcriptome for a specific group. 
			WARNING: Does not accurately model the covariances.

		"""

		cell_selector = (self.anndata.obs[self.group_label] == group) if group != 'all' else np.arange(self.anndata.shape[0])

		observed = self.anndata.X[cell_selector.values, :]
		N = observed.shape[0]

		continuous_normal = np.random.multivariate_normal(
			mean=self.estimated_mus[group],
			cov=self.estimated_sigmas[group],
			size=N)

		# If the count is exactly 0, this was generated from 0 mean 0 variance counts. The final expression should be zero.
		#continuous_normal[continuous_normal == 0.00] -= 100

		# Convert to lognormal
		continuous_lognormal = np.exp(continuous_normal)

		# Round and sample
		return np.random.binomial(np.round(continuous_lognormal).astype(np.int64), p=self.p)


	def differential_expression(self, group_1, group_2):
		"""
			Perform transcriptome wide differential expression analysis between two groups of cells.
		"""

		cell_selector_1 = (self.anndata.obs[self.group_label] == group_1) if group_1 != 'all' else np.arange(self.anndata.shape[0])
		observed_1 = self.anndata.X[cell_selector_1.values, :]
		N_1 = observed_1.shape[0]

		cell_selector_2 = (self.anndata.obs[self.group_label] == group_2) if group_2 != 'all' else np.arange(self.anndata.shape[0])
		observed_2 = self.anndata.X[cell_selector_2.values, :]
		N_2 = observed_2.shape[0]

		mu_1 = self.estimated_mus[group_1]
		mu_2 = self.estimated_mus[group_2]

		var_1 = np.diag(self.estimated_sigmas[group_1])
		var_2 = np.diag(self.estimated_sigmas[group_2])

		s_delta_var = (var_1/N_1)+(var_2/N_2)

		t_statistic = (mu_1 - mu_2) / np.sqrt(s_delta_var)

		dof = s_delta_var**2 / (((var_1 / N_1)**2 / (N_1-1))+((var_2 / N_2)**2 / (N_2-1)))

		pval = stats.t.sf(t_statistic, df=dof)*2 * (t_statistic > 0) + stats.t.cdf(t_statistic, df=dof)*2 * (t_statistic <= 0)

		return t_statistic, dof, pval


	def differential_expression_bayes(self, group_1, group_2):
		""" Bayes factor for differential expression. """

		prob = stats.norm.sf(
		    0, 
		    loc=self.estimated_mus[group_1] - self.estimated_mus[group_2],
		    scale=np.sqrt(np.diag(self.estimated_sigmas[group_1]) + np.diag(self.estimated_sigmas[group_2])))

		return np.log(prob / (1-prob)) * (self.estimated_mus[group_1] != 0.0) * (self.estimated_mus[group_2] != 0.0)


	def differential_variance(self, group_1, group_2, method='f-test'):
		"""
			Perform transcriptome wide differential variance analysis between two groups of cells
		"""

		cell_selector_1 = (self.anndata.obs[self.group_label] == group_1) if group_1 != 'all' else np.arange(self.anndata.shape[0])
		observed_1 = self.anndata.X[cell_selector_1.values, :]
		N_1 = observed_1.shape[0]

		cell_selector_2 = (self.anndata.obs[self.group_label] == group_2) if group_2 != 'all' else np.arange(self.anndata.shape[0])
		observed_2 = self.anndata.X[cell_selector_2.values, :]
		N_2 = observed_2.shape[0]

		var_1 = np.diag(self.estimated_sigmas[group_1])
		var_2 = np.diag(self.estimated_sigmas[group_2])

		if method == 'f-test':

			variance_ratio = var_1 / var_2

			return variance_ratio, stats.f.sf(variance_ratio, N_1, N_2)*2 * (variance_ratio > 1) + stats.f.cdf(variance_ratio, N_1, N_2)*2 * (variance_ratio <= 1)

		elif method == 'levene':

			print('Not implemented yet!')
			return

		else:

			print('Please pick an available option!')
			return


	def differential_variance_bayes(self, group_1, group_2, method='f-test'):
		"""
			Perform transcriptome wide differential variance analysis between two groups of cells in a Bayesian fashion.
		"""

		rotation_mat = np.array([[np.cos(-np.pi/4), -np.sin(-np.pi/4)], [np.sin(-np.pi/4), np.cos(-np.pi/4)]])

		var_1 = np.diag(self.estimated_sigmas[group_1])
		var_2 = np.diag(self.estimated_sigmas[group_2])

		prob = np.zeros(var_1.shape[0])

		idx = 0
		for v1, v2 in zip(var_1, var_2):

			if np.isnan(v1) or np.isnan(v2) or not (v1 > 0.0 and v2 > 0.0):
				prob[idx] = 0.5
				idx += 1
				continue

			cov_mat = np.array([[v1, 0], [0, v2]])
			cov_mat_rotated = rotation_mat.dot(cov_mat).dot(rotation_mat.T)
			prob[idx] = 2*stats.multivariate_normal.cdf([0, 0], mean=[0, 0], cov=cov_mat_rotated)
			idx += 1

		return prob, np.log(prob / (1-prob))


	def differential_correlation(self, group_1, group_2, method='pearson'):
		"""
			Differential correlation using Pearson's method.
		"""

		cell_selector_1 = (self.anndata.obs[self.group_label] == group_1) if group_1 != 'all' else np.arange(self.anndata.shape[0])
		observed_1 = self.anndata.X[cell_selector_1.values, :]
		N_1 = observed_1.shape[0]

		cell_selector_2 = (self.anndata.obs[self.group_label] == group_2) if group_2 != 'all' else np.arange(self.anndata.shape[0])
		observed_2 = self.anndata.X[cell_selector_2.values, :]
		N_2 = observed_2.shape[0]

		if method == 'pearson':
			corr_1 = cov2corr(self.estimated_sigmas[group_1])
			corr_2 = cov2corr(self.estimated_sigmas[group_2])

			z_1 = (1/2) * (np.log(1+corr_1) - np.log(1-corr_1))
			z_2 = (1/2) * (np.log(1+corr_2) - np.log(1-corr_2))

			dz = (z_1 - z_2) / np.sqrt(np.absolute((1/(N_1-3))**2 - (1/(N_2-3))**2))

			return (z_1 - z_2), 2*stats.norm.sf(dz) * (dz > 0) + 2*stats.norm.cdf(dz) * (dz <= 0)

		else:

			print('Not yet implemented!')
			return









