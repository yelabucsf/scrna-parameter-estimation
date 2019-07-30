"""
	simplesc.py
	This file contains code for implementing the empirical bayes estimator for the Gaussian assumption for true single cell RNA sequencing counts.
"""


import pandas as pd
import scipy.stats as stats
import numpy as np
import time
import itertools
import scipy as sp
import time
import logging
from scipy.stats import multivariate_normal
import pickle as pkl
from statsmodels.stats.moment_helpers import cov2corr
from statsmodels.stats.multitest import fdrcorrection
import sklearn.decomposition as decomp

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


class SingleCellEstimator(object):
	"""
		SingleCellEstimator is the class for fitting univariate and bivariate single cell data. 
	"""


	def __init__(
		self, 
		adata,
		group_label,
		n_umis_column,
		num_permute=10000,
		p=0.1):

		self.anndata = adata.copy()
		self.genes = adata.var.index.values
		self.barcodes = adata.obs.index
		self.p = p
		self.group_label = group_label
		self.group_counts = dict(adata.obs[group_label].value_counts())
		for group in list(self.group_counts.keys()):
			self.group_counts['-' + group] = self.anndata.shape[0] - self.group_counts[group]
		self.n_umis = adata.obs[n_umis_column].values
		self.num_permute = num_permute

		# Initialize parameter containing dictionaries
		self.observed_moments = {}
		self.estimated_moments = {}
		self.estimated_central_moments = {}
		self.parameters = {}
		self.parameters_confidence_intervals = {}

		# dictionaries for hypothesis testing
		self.hypothesis_test_result_1d = {}
		self.hypothesis_test_result_2d = {}


	def _compute_statistics(self, observed, N):
		""" Compute some non central moments of the observed data. """

		# Turn the highest value into a 0
		observed[observed == observed.max()] = 0

		if type(observed) == sp.sparse.csr_matrix or type(observed) == sp.sparse.csc.csc_matrix:

			mean = observed.mean(axis=0).A1
			cov = ((observed.T*observed -(sum(observed).T*sum(observed)/N))/(N-1)).todense()

		else:

			mean = observed.mean(axis=0)
			cov = np.cov(observed, rowvar=False)

		prod_expect = cov + mean.reshape(-1,1).dot(mean.reshape(1, -1))

		# Return the first moment, second noncentral moment, expectation of the product
		return mean, np.diag(prod_expect), prod_expect


	def _select_cells(self, group):
		""" Select the cells. """

		if group == 'all': # All cells
			cell_selector = np.arange(self.anndata.shape[0])
		elif group[0] == '-': # Exclude this group
			cell_selector = (self.anndata.obs[self.group_label] != group[1:]).values
		else: # Include this group
			cell_selector = (self.anndata.obs[self.group_label] == group).values

		return cell_selector


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


	def compute_params(self, group='all'):
		""" 
			Use the estimated moments to compute the parameters of marginal distributions as 
			well as the estimated correlation. 
		"""

		self.parameters[group] = {
			'mean': self.estimated_central_moments[group]['first'],
			'dispersion': self.estimated_central_moments[group]['second'] / self.estimated_central_moments[group]['first'],
			'corr': cov2corr(self.estimated_central_moments[group]['prod'])}

		self.parameters[group]['log_mean'] = np.log(self.parameters[group]['mean'])
		self.parameters[group]['log_dispersion'] = np.log(self.parameters[group]['dispersion'])


	def compute_confidence_intervals(self, group='all'):
		"""
			Compute 95% confidence intervals around the estimated parameters. 

			Use the Gamma -> Dirichlet framework to speed up the process.

			Uses self.num_permute attribute, same as in hypothesis testing.

			CAVEAT: Uses the same expectation of 1/N as the true value, does not compute this from the permutations. 
			So the result might be slightly off.
		"""

		mean_inv_numis = self.p * self.observed_moments[group]['allgenes_second'] / self.observed_moments[group]['allgenes_first']**3

		cell_selector = self._select_cells(group)
		data = self.anndata.X[cell_selector, :].toarray()

		all_counts = set()
		gene_counts = []

		# Collect all unique counts in this group
		for gene_idx in range(self.anndata.var.shape[0]):

			hist = np.bincount(data[:, gene_idx].reshape(-1).astype(int))
			gene_counts.append(hist)
			all_counts |= set(hist)
		all_counts = np.array(sorted(list(all_counts)))

		# Define the gamma variales to be later used to construct the Dirichlet
		gamma_rvs = stats.gamma.rvs(
			a=(all_counts+1e-10), 
			size=(self.num_permute, all_counts.shape[0]))

		# Declare placeholders for gene confidence intervals
		mean_conf_interval = np.zeros(self.anndata.var.shape[0])
		dispersion_conf_interval = np.zeros(self.anndata.var.shape[0])
		log_mean_conf_interval = np.zeros(self.anndata.var.shape[0])
		log_dispersion_conf_interval = np.zeros(self.anndata.var.shape[0])
		log1p_mean_conf_interval = np.zeros(self.anndata.var.shape[0])
		log1p_dispersion_conf_interval = np.zeros(self.anndata.var.shape[0])

		# Iterate through each gene and compute a p value for each gene
		for gene_idx in range(self.anndata.var.shape[0]):

			# Grab the appropriate Gamma variables given the bincounts of this particular gene
			gene_gamma_rvs = gamma_rvs[:, np.nonzero(gene_counts[gene_idx][:, None] == all_counts)[1]]

			# Sample dirichlet from the Gamma variables
			gene_dir_rvs = gene_gamma_rvs/gene_gamma_rvs.sum(axis=1)[:,None]

			# Construct the repeated values matrix
			values = np.tile(np.arange(0, gene_counts[gene_idx].shape[0]).reshape(1, -1), (self.num_permute, 1))

			# Compute the permuted, observed mean/dispersion for group 1
			mean = ((gene_dir_rvs) * values).sum(axis=1)
			second_moments = ((gene_dir_rvs) * values**2).sum(axis=1)

			# Compute the permuted, estimated moments for both groups
			moment_map = np.linalg.inv(np.array([[self.p, 0], [self.p - self.p**2, self.p**2 - self.p*mean_inv_numis]]))
			estimated_moments = moment_map.dot(np.vstack([mean, second_moments]))

			# Compute the permuted parameters
			estimated_means = estimated_moments[0, :]
			estimated_dispersions = (estimated_moments[1, :] - estimated_moments[0, :]**2)/estimated_moments[0, :]

			# Store the S.E. of the parameter, log(param), and log1p(param)
			mean_conf_interval[gene_idx] = np.nanstd(estimated_means)
			dispersion_conf_interval[gene_idx] = np.nanstd(estimated_dispersions)
			log_mean_conf_interval[gene_idx] = np.nanstd(np.log(estimated_means))
			log_dispersion_conf_interval[gene_idx] = np.nanstd(np.log(estimated_dispersions))
			log1p_mean_conf_interval[gene_idx] = np.nanstd(np.log(estimated_means+1))
			log1p_dispersion_conf_interval[gene_idx] = np.nanstd(np.log(estimated_dispersions+1))

		# Update the attribute dictionary
		self.parameters_confidence_intervals[group] = {
			'mean':mean_conf_interval,
			'dispersion':dispersion_conf_interval,
			'log_mean':log_mean_conf_interval,
			'log_dispersion':log_dispersion_conf_interval,
			'log1p_mean':log1p_mean_conf_interval,
			'log1p_dispersion':log1p_dispersion_conf_interval}


	def _compute_pval(self, statistic, null_statistics, method='two-tailed'):
		""" 
			Compute empirical pvalues from the given null statistics. To protect against asymmetric distributions, double the smaller one sided statistic.
		"""

		if method == 'two-tailed':

			median_null = np.nanmedian(null_statistics)
			abs_t = np.absolute(median_null - statistic)
			pval = ((null_statistics > abs_t).mean() + (null_statistics < -abs_t).mean()) if not np.isnan(abs_t) else np.nan

		elif method == 'one-tailed':

			#pval = np.array([(null_statistics[null_statistics > 0] > s).mean()*(s > 0) + (null_statistics[null_statistics < 0] < s).mean()*(s < 0) if not np.isnan(s) else np.nan for s in statistics])

			pval = 2*min((null_statistics > statistic).mean(), (null_statistics < statistic).mean()) if not np.isnan(statistic) else np.nan

		return pval


	def _fdrcorrect(self, pvals):
		"""
			Perform FDR correction with nan's.
		"""

		fdr = np.ones(pvals.shape[0])
		_, fdr[~np.isnan(pvals)] = fdrcorrection(pvals[~np.isnan(pvals)])
		return fdr


	def hypothesis_test_1d(self, group_1, group_2):
		"""
			Compute a p value via permutation testing for difference in mean and difference in dispersion.

			This function also holds code for fast permutation testing for discrete data via independent sampling 
			of Gamma, then turning it into Dirichlet.
			TODO: Refactor this gross monstrosity.
		"""

		# Define the group key
		group_key = frozenset([group_1, group_2])

		# Stop if we already did this hypothesis test
		if group_key in self.hypothesis_test_result_1d:
			print('Already computed 1d hypothesis test')
			return

		# Set up for computing the null distribution
		cell_selector = self._select_cells(group_1) | self._select_cells(group_2)
		data = self.anndata.X[cell_selector, :].toarray()

		all_counts = set()
		gene_counts = []

		mean_inv_numis = self.p * (self.n_umis[cell_selector]**2).mean() / self.n_umis[cell_selector].mean()**3

		# Get possible unique counts from all the genes to pass onto the Gamma distribution
		for gene_idx in range(self.anndata.var.shape[0]):

			hist = np.bincount(data[:, gene_idx].reshape(-1).astype(int))
			gene_counts.append(hist)
			all_counts |= set(hist)
		all_counts = np.array(sorted(list(all_counts)))

		# Generate the gamma random variables to turn into Dirichlet
		gamma_rvs_1 = stats.gamma.rvs(
			a=(all_counts+1e-10)*(self.group_counts[group_1] / (self.group_counts[group_1] + self.group_counts[group_2])), 
			size=(self.num_permute, all_counts.shape[0]))
		gamma_rvs_2 = stats.gamma.rvs(
			a=(all_counts+1e-10)*(self.group_counts[group_2] / (self.group_counts[group_1] + self.group_counts[group_2])), 
			size=(self.num_permute, all_counts.shape[0]))

		# Get the test statistics
		de_diff = self.parameters[group_2]['log_mean'] - self.parameters[group_1]['log_mean']
		dv_diff = self.parameters[group_2]['log_dispersion'] - self.parameters[group_1]['log_dispersion']

		# Declare placeholders for the pvalues
		de_pvals = np.zeros(self.anndata.var.shape[0])
		dv_pvals = np.zeros(self.anndata.var.shape[0])

		# Iterate through each gene and compute a p value for each gene
		for gene_idx in range(self.anndata.var.shape[0]):

			# Grab the appropriate Gamma variables given the bincounts of this particular gene
			gene_gamma_rvs_1 = gamma_rvs_1[:, np.nonzero(gene_counts[gene_idx][:, None] == all_counts)[1]]
			gene_gamma_rvs_2 = gamma_rvs_2[:, np.nonzero(gene_counts[gene_idx][:, None] == all_counts)[1]]

			# Sample dirichlet from the Gamma variables
			gene_dir_rvs_1 = gene_gamma_rvs_1/gene_gamma_rvs_1.sum(axis=1)[:,None]
			gene_dir_rvs_2 = gene_gamma_rvs_2/gene_gamma_rvs_2.sum(axis=1)[:,None]

			values = np.tile(np.arange(0, gene_counts[gene_idx].shape[0]).reshape(1, -1), (self.num_permute, 1))

			# Compute the permuted, observed mean/dispersion for group 1
			mean_1 = ((gene_dir_rvs_1) * values).sum(axis=1)
			second_moments_1 = ((gene_dir_rvs_1) * values**2).sum(axis=1)

			# Compute the permuted, observed mean/dispersion for group 2
			mean_2 = ((gene_dir_rvs_2) * values).sum(axis=1)
			second_moments_2 = ((gene_dir_rvs_2) * values**2).sum(axis=1)

			# Compute the permuted, estimated moments for both groups
			moment_map = np.linalg.inv(np.array([[self.p, 0], [self.p - self.p**2, self.p**2 - self.p*mean_inv_numis]]))
			moments_1 = moment_map.dot(np.vstack([mean_1, second_moments_1]))
			moments_2 = moment_map.dot(np.vstack([mean_2, second_moments_2]))

			# Compute the null (permuted) test statistics
			null_mean = np.log(moments_2[0, :]) - np.log(moments_1[0, :])
			null_dispersion = np.log((moments_2[1, :] - moments_2[0, :]**2)/moments_2[0, :]) - np.log((moments_1[1, :] - moments_1[0, :]**2)/moments_1[0, :])

			# Compute the p-values
			de_pvals[gene_idx] = self._compute_pval(
				statistic=de_diff[gene_idx],
				null_statistics=null_mean)
			dv_pvals[gene_idx] = self._compute_pval(
				statistic=dv_diff[gene_idx],
				null_statistics=null_dispersion)

		# Perform FDR correction and save the result
		self.hypothesis_test_result_1d[group_key] = {
			'de_diff': de_diff,
			'de_pval': de_pvals,
			'de_fdr': self._fdrcorrect(de_pvals),
			'dv_diff': dv_diff,
			'dv_pval': dv_pvals,
			'dv_fdr': self._fdrcorrect(dv_pvals),		
		}


	def hypothesis_test_2d(self, group_1, group_2, gene_list_1, gene_list_2):
		""" Comparison of correlation between two groups. """

		# Define the group key
		group_key = frozenset([group_1, group_2])

		# Stop if we already did this hypothesis test
		if group_key in self.hypothesis_test_result_2d:
			print('Already computed 1d hypothesis test')
			return

		# Set up for computing the null distribution
		cell_selector = self._select_cells(group_1) | self._select_cells(group_2)
		data = self.anndata.X[cell_selector, :].toarray()

		all_pair_counts = set()
		pair_counts = {}

		# Compute the mean_inv_numis
		mean_inv_numis = self.p * (self.n_umis[cell_selector]**2).mean() / self.n_umis[cell_selector].mean()**3

		# Get the gene idx for the two gene lists
		genes_idxs_1 = np.array([np.where(self.anndata.var.index == gene)[0][0] for gene in gene_list_1])
		genes_idxs_2 = np.array([np.where(self.anndata.var.index == gene)[0][0] for gene in gene_list_2])

		for gene_idx_1 in genes_idxs_1:
			pair_counts[gene_idx_1] = {}

			for gene_idx_2 in genes_idxs_2:
				cantor_code = pair(data[:, gene_idx_1], data[:, gene_idx_2])
				hist = np.bincount(cantor_code)
				all_pair_counts |= set(hist)
				pair_counts[gene_idx_1][gene_idx_2] = hist
		all_pair_counts = np.array(sorted(list(all_pair_counts)))

		# Generate the gamma random variables to turn into Dirichlet
		gamma_rvs_1 = stats.gamma.rvs(
			a=(all_pair_counts+1e-10)*(self.group_counts[group_1] / (self.group_counts[group_1] + self.group_counts[group_2])), 
			size=(self.num_permute, all_pair_counts.shape[0]))
		gamma_rvs_2 = stats.gamma.rvs(
			a=(all_pair_counts+1e-10)*(self.group_counts[group_2] / (self.group_counts[group_1] + self.group_counts[group_2])), 
			size=(self.num_permute, all_pair_counts.shape[0]))
		# Get the test statistics
		dc_diff = self.parameters[group_2]['corr'] - self.parameters[group_1]['corr']

		# Declare placeholders for the pvalues
		dc_pvals = np.zeros(dc_diff.shape)*np.nan

		for gene_idx_1 in genes_idxs_1:
			for gene_idx_2 in genes_idxs_2:

				if gene_idx_1 == gene_idx_2:
					dc_pvals[gene_idx_1, gene_idx_2] = 1
					continue

				start = time.time()

				# Grab the appropriate Gamma variables given the bincounts of this particular gene
				gene_gamma_rvs_1 = gamma_rvs_1[:, np.nonzero(pair_counts[gene_idx_1][gene_idx_2][:, None] == all_pair_counts)[1]]
				gene_gamma_rvs_2 = gamma_rvs_2[:, np.nonzero(pair_counts[gene_idx_1][gene_idx_2][:, None] == all_pair_counts)[1]]

				# Sample dirichlet from the Gamma variables
				gene_dir_rvs_1 = gene_gamma_rvs_1/gene_gamma_rvs_1.sum(axis=1)[:,None]
				gene_dir_rvs_2 = gene_gamma_rvs_2/gene_gamma_rvs_2.sum(axis=1)[:,None]

				cantor_code = np.arange(0, pair_counts[gene_idx_1][gene_idx_2].shape[0])
				values_1, values_2 = depair(cantor_code)

				values_1 = np.tile(values_1.reshape(1, -1), (self.num_permute, 1))
				values_2 = np.tile(values_2.reshape(1, -1), (self.num_permute, 1))

				# Compute the permuted statistics for group 1
				mean_gene_1_1 = ((gene_dir_rvs_1) * values_1).sum(axis=1)
				second_moments_gene_1_1 = ((gene_dir_rvs_1) * values_1**2).sum(axis=1)
				mean_gene_2_1 = ((gene_dir_rvs_1) * values_2).sum(axis=1)
				second_moments_gene_2_1 = ((gene_dir_rvs_1) * values_2**2).sum(axis=1)
				prod_1 = ((gene_dir_rvs_1) * values_1*values_2).sum(axis=1)

				moment_map = np.linalg.inv(np.array([[self.p, 0], [self.p - self.p**2, self.p**2 - self.p*mean_inv_numis]]))
				estimated_moments_gene_1_1 = moment_map.dot(np.vstack([mean_gene_1_1, second_moments_gene_1_1]))
				estimated_moments_gene_2_1 = moment_map.dot(np.vstack([mean_gene_2_1, second_moments_gene_2_1]))
				estimated_prod_1 = prod_1 / (self.p**2 - self.p*(1-self.p)*mean_inv_numis)
				estimated_corr_1 = \
					(estimated_prod_1 - estimated_moments_gene_1_1[0, :]*estimated_moments_gene_2_1[0, :])/ \
					np.sqrt(
						(estimated_moments_gene_1_1[1, :] - estimated_moments_gene_1_1[0, :]**2) * \
						(estimated_moments_gene_2_1[1, :] - estimated_moments_gene_2_1[0, :]**2))

				# Compute the permuted statistics for group 1
				mean_gene_1_2 = ((gene_dir_rvs_2) * values_1).sum(axis=1)
				second_moments_gene_1_2 = ((gene_dir_rvs_2) * values_1**2).sum(axis=1)
				mean_gene_2_2 = ((gene_dir_rvs_2) * values_2).sum(axis=1)
				second_moments_gene_2_2 = ((gene_dir_rvs_2) * values_2**2).sum(axis=1)
				prod_2 = ((gene_dir_rvs_2) * values_1*values_2).sum(axis=1)

				estimated_moments_gene_1_2 = moment_map.dot(np.vstack([mean_gene_1_2, second_moments_gene_1_2]))
				estimated_moments_gene_2_2 = moment_map.dot(np.vstack([mean_gene_2_2, second_moments_gene_2_2]))
				estimated_prod_2 = prod_2 / (self.p**2 - self.p*(1-self.p)*mean_inv_numis)
				estimated_corr_2 = \
					(estimated_prod_2 - estimated_moments_gene_1_2[0, :]*estimated_moments_gene_2_2[0, :])/ \
					np.sqrt(
						(estimated_moments_gene_1_2[1, :] - estimated_moments_gene_1_2[0, :]**2) * \
						(estimated_moments_gene_2_2[1, :] - estimated_moments_gene_2_2[0, :]**2))

				# Compute the null (permuted) test statistics
				null_corr_diff = estimated_corr_2 - estimated_corr_1

				# Compute the p-values
				dc_pvals[gene_idx_1, gene_idx_2] = self._compute_pval(
					statistic=dc_diff[gene_idx_1, gene_idx_2],
					null_statistics=null_corr_diff)

		# Perform FDR correction
		dc_fdr = dc_pvals.copy()
		fdr = self._fdrcorrect(dc_pvals[genes_idxs_1, :][:, genes_idxs_2].reshape(-1))\
			.reshape(genes_idxs_1.shape[0], genes_idxs_2.shape[0])
		for idx1, gene_idx_1 in enumerate(genes_idxs_1):
			for idx2, gene_idx_2 in enumerate(genes_idxs_2):
				dc_fdr[gene_idx_1, gene_idx_2] = fdr[idx1, idx2]

		# Perform FDR correction and save the result
		self.hypothesis_test_result_2d[group_key] = {
			'dc_diff': dc_diff[genes_idxs_1, :][:, genes_idxs_2],
			'gene_idx_1': genes_idxs_1,
			'gene_idx_2': genes_idxs_2,
			'dc_pval': dc_pvals[genes_idxs_1, :][:, genes_idxs_2],
			'dc_fdr': dc_fdr[genes_idxs_1, :][:, genes_idxs_2],		
		}


	def get_differential_genes(self, group_1, group_2, which, direction, sig=0.1, num_genes=50):
		"""
			Get genes that are increased in expression in group 2 compared to group 1, sorted in order of significance.
			:which: should be either "mean" or "dispersion"
			:direction: should be either "increase" or "decrease"
			:sig: defines the threshold
			:num_genes: defines the number of genes to be returned. If bigger than the number of significant genes, then return only the significant ones.
		"""

		# Setup keys
		group_key = frozenset([group_1, group_2])
		param_key = 'de' if which == 'mean' else 'dv'

		# Find the number of genes to return
		sig_condition = self.hypothesis_test_result_1d[group_key][param_key + '_fdr'] < sig
		dir_condition = ((1 if direction == 'increase' else -1)*self.hypothesis_test_result_1d[group_key][param_key + '_diff']) > 0
		num_sig_genes = (sig_condition & dir_condition).sum()
		
		# We will order the output by significance. Just turn the FDR of the other half into 1's to remove them from the ordering.
		relevant_fdr = self.hypothesis_test_result_1d[group_key][param_key + '_fdr'].copy()
		relevant_fdr[~dir_condition] = 1

		# Get the order of the genes in terms of FDR.
		order = np.argsort(relevant_fdr)[:min(num_sig_genes, num_genes)]

		return relevant_fdr[order], self.genes[order]
