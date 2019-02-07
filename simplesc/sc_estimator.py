"""
	sc_estimator.py
	This file contains code for fitting 1d and 2d lognormal and normal parameters to scRNA sequencing count data.
"""


import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
from statsmodels.stats.weightstats import DescrStatsW
import numpy as np
import itertools
import logging
from scipy.stats import multivariate_normal


class SingleCellEstimator(object):
	"""
		SingleCellEstimator is the class for fitting univariate and bivariate single cell data. 
	"""

	def __init__(
		self, 
		adata, 
		p=0.1, 
		group_label='leiden'):

		self.anndata = adata
		self.genes = adata.var.index
		self.barcodes = adata.obs.index
		self.p = p
		self.group_label = group_label
		self.param_1d = {}
		self.param_2d = {}
		self.fitting_progress = {}
		self.diff_exp = {}

		# TODO: Define a heuristic on the max latent based on observed counts
		self.max_latent=50


	def _fit_lognormal(self, x, w=None):
		""" Fit a weighted 1d lognormal. """

		lnx = np.log(x[x > 0])
		muhat = np.average(lnx, weights=w[x > 0])
		varhat = np.average((lnx - muhat)**2, weights=w[x > 0])

		return muhat, np.sqrt(varhat)


	def _rv_pmf(self, x, mu, sigma):
		""" PDF/PMF of the random variable under use. """

		return stats.lognorm.cdf(0.5, s=sigma, loc=0, scale=np.exp(mu))*(x==0) + \
			(stats.lognorm.cdf(x+0.5, s=sigma, loc=0, scale=np.exp(mu)) - stats.lognorm.cdf(x-0.5, s=sigma, loc=0, scale=np.exp(mu)))*(x!=0)


	def _create_px_table(self, mu, sigma, p):
		return np.array([
			(self._rv_pmf(np.arange(x, self.max_latent), mu, sigma) * stats.binom.pmf(x, np.arange(x, self.max_latent), p)).sum()
			for x in range(self.max_latent)])


	def _create_pz_table(self, mu, sigma, p):
		""" Returns a matrix M x M where rows indicate X and columns indicate Z """

		px_table = self._create_px_table(mu, sigma, p)

		table = np.zeros(shape=(self.max_latent, self.max_latent))

		for x in range(self.max_latent):
			for z in range(x, self.max_latent):
				table[x, z] = self._rv_pmf(z, mu, sigma) * stats.binom.pmf(x, z, p) / px_table[x]
		return table

	def _get_parameters(self, observed, prob_table, p_hat):
		""" Get the parameters of the Gaussian and dropout. """

		data = pd.DataFrame()
		data['observed'] = observed
		data = data.groupby('observed').size().reset_index(name='count')
		data['observed_weight'] = data['count'] / len(observed)
		data = data.merge(
		pd.concat(
			[pd.DataFrame(
				np.concatenate(
					[np.ones(self.max_latent-x).reshape(-1, 1)*x, 
					np.arange(x, self.max_latent).reshape(-1,1),
					prob_table[x, x:].reshape(-1, 1)], axis=1), 
				columns=['observed', 'latent', 'latent_weight']) for x in range(self.max_latent)]),
		on='observed', 
		how='left')
		data['point_weight'] = data['observed_weight'] * data['latent_weight']
		data['p_estimates'] = (data['observed'] / data['latent'] * data['point_weight']).fillna(0.0).replace(np.inf, 0.0)
		p_estimate = p_hat #min(max(data['p_estimates'].sum(), 0.05), 0.15)

		mu, sigma = self._fit_lognormal(data['latent'].values, w=data['point_weight'].values)

		return mu, sigma, p_estimate


	def _lognormal_pdf_2d(self, x, mean, cov):

		denom = x[0]*x[1] if type(x) == list else (x[:, 0] * x[:, 1])
		return np.nan_to_num(stats.multivariate_normal.pdf(np.log(x), mean=mean, cov=cov)/denom)


	def _multivariate_pdf(self, x, mean, cov, method='lognormal'):
		""" Multivariate normal PMF. """

		if method == 'normal':

			return multivariate_normal.pdf(x, mean=mean, cov=cov)

		if method == 'lognormal':

			return self._lognormal_pdf_2d(x, mean=mean, cov=cov)


	def _calculate_px_2d(self, x1, x2, mu, sigma, p, method, max_count=21):
		z_candidates = np.array(list(itertools.product(np.arange(x1, max_count), np.arange(x2, max_count))))
		return (
			stats.binom.pmf(x1, z_candidates[:, 0], p) * \
			stats.binom.pmf(x2, z_candidates[:, 1], p) * \
			self._multivariate_pdf(z_candidates, mean=mu, cov=sigma, method=method)
			).sum()


	def _create_px_table_2d(self, mu, sigma, p, method, max_count=21):

		px_table = np.zeros((max_count, max_count))
		for i in range(max_count):
			for j in range(max_count):
				px_table[i][j] = self._calculate_px_2d(i, j, mu, sigma, p, method)
		return px_table


	def compute_1d_params(
		self, 
		gene, 
		group='all', 
		initial_p_hat=0.1, 
		initial_mu_hat=5, 
		initial_sigma_hat=1, 
		num_iter=200):
		""" Compute 1d params for a gene for a specific group. """

		# Initialize the parameters
		mu_hat = initial_mu_hat
		sigma_hat = initial_sigma_hat
		p_hat = initial_p_hat

		# Select the data
		cell_selector = (self.anndata.obs[self.group_label] == group) if group != 'all' else np.arange(self.anndata.shape[0])
		gene_selector = self.anndata.var.index.get_loc(gene)
		observed = self.anndata.X[cell_selector.values, gene_selector]
		if type(self.anndata.X) != np.ndarray:
			observed = observed.toarray().reshape(-1)
		observed_counts = pd.Series(observed).value_counts()

		# Fitting progress
		fitting_progress = []
		for itr in range(num_iter):

			# Compute the log likelihood
			px_table = self._create_px_table(mu_hat, sigma_hat, p_hat)
			log_likelihood = np.array([count*np.log(px_table[int(val)]) for val,count in zip(observed_counts.index, observed_counts)]).sum()
			

			# Early stopping and prevent pathological behavior
			stopping_condition = \
				itr > 0 and \
				(np.isinf(log_likelihood) or \
					np.isnan(log_likelihood) or \
					(log_likelihood < fitting_progress[-1][4] + 1e-4) or \
					np.isnan(mu_hat) or \
					np.isnan(sigma_hat))

			if stopping_condition:
				mu_hat = fitting_progress[-1][1]
				sigma_hat = fitting_progress[-1][2]
				break

			# Keep track of the fitting progress
			fitting_progress.append((itr, mu_hat, sigma_hat, p_hat, log_likelihood))

			# E step
			prob_table = self._create_pz_table(mu_hat, sigma_hat, p_hat)

			# M step
			mu_hat, sigma_hat, p_hat = self._get_parameters(observed, prob_table, p_hat)

		fitting_progress = pd.DataFrame(fitting_progress, columns=['iteration', 'mu_hat', 'sigma_hat', 'p_hat', 'log_likelihood'])

		if gene not in self.param_1d:
			self.param_1d[gene] = {}
			self.fitting_progress[gene] = {}
		self.param_1d[gene][group] = (mu_hat, sigma_hat, p_hat, observed.shape[0])
		self.fitting_progress[gene][group] = fitting_progress.copy()


	def generate_reconstructed_obs(self, gene, group='all'):
		""" Generate a dropped out distribution for qualitative check. """

		mu_hat, sigma_hat, p_hat, n = self.param_1d[gene][group]
		reconstructed_counts = np.random.binomial(n=np.round(stats.lognorm.rvs(s=sigma_hat, scale=np.exp(mu_hat), size=n)).astype(np.int64), p=p_hat)

		return reconstructed_counts


	def differential_expression(self, gene, groups, method='t-test'):
		"""
			Performs Welch's t-test for unequal variances between two groups for the same gene.
			Groups should be a tuple of different names.

			Users are encouraged to keep alphabetical ordering in groups.
		"""

		if not (gene in self.param_1d and groups[0] in self.param_1d[gene] and groups[1] in self.param_1d[gene]):
			raise 'Please fit the parameters first!'

		# Get the parameters
		mu_1, sigma_1, p_1, n_1 = self.param_1d[gene][groups[0]]
		mu_2, sigma_2, p_2, n_2 = self.param_1d[gene][groups[1]]

		if method == 't-test':

			s_delta_var = (sigma_1**2/n_1)+(sigma_2**2/n_2)
			t_statistic = (mu_1 - mu_2) / np.sqrt(s_delta_var)
			dof = s_delta_var**2 / (((sigma_1**2 / n_1)**2 / (n_1-1))+((sigma_2**2 / n_2)**2 / (n_2-1)))
			pval = stats.t.sf(t_statistic, df=dof)*2 if t_statistic > 0 else stats.t.cdf(t_statistic, df=dof)*2

			if gene not in self.diff_exp:
				self.diff_exp[gene] = {}
			self.diff_exp[gene][groups] = (t_statistic, dof, pval)

		else:

			raise 'Not implemented!'





