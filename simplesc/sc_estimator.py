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
import time
import logging
from scipy.stats import multivariate_normal
import pickle as pkl


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
		self.param_sweep_2d = {}
		self.log_likelihood_2d = {}
		self.fitting_progress = {}
		self.fitting_progress_2d = {}
		self.diff_exp = {}

		# TODO: Define a heuristic on the max latent based on observed counts
		self.max_latent = 100
		self.max_latent_2d = 60

		# TODO: Find optimal zero padding size
		self.zero_pad_size = 100000

		# Attributes for 1D/2D estimation
		self._create_binom_table()
		self._create_base_dataframe()

		# Attributes for 2D estimation
		self.z_candidates_2d = np.array(list(itertools.product(np.arange(0, self.max_latent_2d), np.arange(0, self.max_latent_2d))))
		self.d1_binom_probs = np.tile(self.binom_table[:self.max_latent_2d, :self.max_latent_2d], self.max_latent_2d)
		self.d2_binom_probs = np.repeat(self.binom_table[:self.max_latent_2d, :self.max_latent_2d], self.max_latent_2d, axis=1)

		self.z_cand_rep = np.tile(self.z_candidates_2d, (self.max_latent_2d**2, 1))
		self.x_table_rep = np.repeat(self.z_candidates_2d, self.max_latent_2d**2, axis=0)
		self.d1_binom_rep = np.repeat(self.binom_table[:self.max_latent_2d, :self.max_latent_2d].reshape(-1, 1), self.max_latent_2d**2, axis=0)
		self.d2_binom_rep = np.tile(self.binom_table[:self.max_latent_2d, :self.max_latent_2d].reshape(-1, 1), (self.max_latent_2d**2, 1))


	def _create_base_dataframe(self):

		self.base_dataframe = pd.concat(
			[pd.DataFrame(
				np.concatenate(
					[np.ones(self.max_latent-x).reshape(-1, 1)*x, 
					np.arange(x, self.max_latent).reshape(-1,1)], axis=1), 
				columns=['observed', 'latent']) for x in range(self.max_latent)])
		self.base_dataframe.set_index('observed', inplace=True)


	def _create_binom_table(self):

		self.binom_table = np.zeros(shape=(self.max_latent, self.max_latent))

		for x in range(self.max_latent):
			for z in range(x, self.max_latent):
				self.binom_table[x, z] = stats.binom.pmf(x, z, self.p)


	def _fit_lognormal(self, x, w=None, neg_mean=False):
		""" Fit a weighted 1d lognormal. """

		if neg_mean:

			x = np.concatenate([
				np.random.uniform(high=0.5, size=self.zero_pad_size),
				x[1:]])
			w = np.concatenate([
				w[0]/self.zero_pad_size * np.ones(self.zero_pad_size),
				w[1:]])

		lnx = np.log(x[x > 0])
		muhat = np.average(lnx, weights=w[x > 0])
		varhat = np.average((lnx - muhat)**2, weights=w[x > 0])

		return muhat, np.sqrt(varhat)


	def _rv_pmf(self, x, mu, sigma):
		""" PDF/PMF of the random variable under use. """

		return stats.lognorm.cdf(0.5, s=sigma, loc=0, scale=np.exp(mu))*(x==0) + \
			(stats.lognorm.cdf(x+0.5, s=sigma, loc=0, scale=np.exp(mu)) - stats.lognorm.cdf(x-0.5, s=sigma, loc=0, scale=np.exp(mu)))*(x!=0)


	def _create_px_table(self, mu, sigma, p):

		z_probs = np.tile(self._rv_pmf(np.arange(0, self.max_latent), mu, sigma).reshape(1, -1), [self.max_latent, 1])

		return (z_probs * self.binom_table).sum(axis=1)


	def _create_pz_table(self, mu, sigma, p, px_table):
		""" Returns a matrix M x M where rows indicate X and columns indicate Z """


		px_table = np.tile(px_table.reshape(-1, 1), [1, self.max_latent])
		z_probs = np.tile(self._rv_pmf(np.arange(0, self.max_latent), mu, sigma).reshape(1, -1), [self.max_latent, 1])
		table = z_probs * self.binom_table / px_table

		return table


	def _get_parameters(self, observed_data, prob_table, p_hat, neg_mean=False):
		""" Get the parameters of the Gaussian and dropout. """

		self.base_dataframe['latent_weight'] = np.concatenate([prob_table[x, x:].reshape(-1, 1) for x in range(self.max_latent)])

		data = observed_data\
			.join(
				self.base_dataframe,
				how='left')

		data['point_weight'] = data['observed_weight'] * data['latent_weight']

		mu, sigma = self._fit_lognormal(data['latent'].values, w=data['point_weight'].values, neg_mean=neg_mean)

		return mu, sigma, p_hat


	def _lognormal_pdf_2d(self, x, mean, cov):

		denom = x[0]*x[1] if type(x) == list else (x[:, 0] * x[:, 1])
		return np.nan_to_num(stats.multivariate_normal.pdf(np.log(x), mean=mean, cov=cov)/denom)


	def _lognormal_pmf_2d(self, x, mean, cov):

		if type(x) == list:
			x = np.array(x).reshape(-1, 2)

		x = x.astype(np.float64)

		upper_right = x + 0.5

		lower_left = x - 0.5

		upper_left = x.copy()
		upper_left[:, 0] = upper_left[:, 0] - 0.5
		upper_left[:, 1] = upper_left[:, 1] + 0.5

		lower_right = x.copy()
		lower_right[:, 0] = lower_right[:, 0] + 0.5
		lower_right[:, 1] = lower_right[:, 1] - 0.5

		both_zero = (x.sum(axis=1) == 0)
		d0_zero = (x[:, 0] == 0) & (x[:, 1] != 0)
		d1_zero = (x[:, 1] == 0) & (x[:, 0] != 0)
		both_nonzero = ~(d0_zero | d1_zero | both_zero)

		upper_right_cdf = stats.multivariate_normal.cdf(np.log(upper_right), mean=mean, cov=cov)
		lower_left_cdf = np.nan_to_num(stats.multivariate_normal.cdf(np.log(lower_left), mean=mean, cov=cov))
		upper_left_cdf = np.nan_to_num(stats.multivariate_normal.cdf(np.log(upper_left), mean=mean, cov=cov))
		lower_right_cdf = np.nan_to_num(stats.multivariate_normal.cdf(np.log(lower_right), mean=mean, cov=cov))

		both_zeros_pmf = upper_right_cdf
		d1_zero_pmf = upper_right_cdf - upper_left_cdf
		d0_zero_pmf = upper_right_cdf - lower_right_cdf
		both_nonzero_pmf = upper_right_cdf - lower_right_cdf - (upper_left_cdf-lower_left_cdf)

		return \
			(both_zeros_pmf * both_zero) + \
			(d1_zero_pmf * d1_zero) + \
			(d0_zero_pmf * d0_zero) + \
			(both_nonzero_pmf * both_nonzero)


	def _multivariate_pmf(self, x, mean, cov, method='lognormal'):
		""" Multivariate normal PMF. """

		if method == 'normal':

			return multivariate_normal.pdf(x, mean=mean, cov=cov)

		if method == 'lognormal':

			return self._lognormal_pmf_2d(x, mean=mean, cov=cov)


	def _fit_multivariate_distribution(self, data, d1_col='z1', d2_col='z2', weights_col='point_weight', method='lognormal'):

		if method == 'normal':

			parameters = DescrStatsW(data[[d1_col, d2_col]], weights=data[weights_col])

		elif method == 'lognormal':

			ln_data = data.query('{} > 0 & {} > 0'.format(d1_col, d2_col))

			mu_1, std_1 = self._fit_lognormal(ln_data[d1_col], ln_data[weights_col])
			mu_2, std_2 = self._fit_lognormal(ln_data[d2_col], ln_data[weights_col])

			print(ln_data[weights_col].sum())
			cov = np.average((ln_data[d1_col] - mu_1) * (ln_data[d2_col] - mu_2), weights=ln_data[weights_col])

		return np.array([mu_1, mu_2]), np.array([[std_1**2, cov], [cov, std_2**2]])


	def _create_px_table_2d(self, mu, sigma, p, method='lognormal'):

		pmf_table = np.tile(self._multivariate_pmf(self.z_candidates_2d, mean=mu, cov=sigma, method=method), (self.max_latent_2d, 1))
		return (self.d1_binom_probs * pmf_table).dot(self.d2_binom_probs.T)


	def _create_pz_table_2d(self, mu, sigma, p, px_table, method='lognormal'):
		""" Returns a matrix M x M where rows indicate X and columns indicate Z """

		z_prob = np.tile(self._multivariate_pmf(self.z_candidates_2d, mean=mu, cov=sigma, method=method), (1, self.max_latent_2d**2)).reshape(-1, 1)
		x_prob = np.repeat(px_table.reshape(-1, 1), self.max_latent_2d**2, axis=0)
		table = np.hstack([self.x_table_rep, self.z_cand_rep, z_prob*self.d1_binom_rep*self.d2_binom_rep/x_prob])

		return pd.DataFrame(table, columns=['x1', 'x2', 'z1', 'z2', 'latent_weight'])


	def _get_parameters_2d(self, observed_data, prob_table, initial_p_hat=0.1, neg_mean=False):
		""" Get the parameters of the Gaussian and dropout """

		data = observed_data.merge(
			prob_table,
			on=['x1', 'x2'],
			how='left')

		data['point_weight'] = data['observed_weight'] * data['latent_weight']

		print(data['point_weight'].sum())

		mu, sigma = self._fit_multivariate_distribution(data, d1_col='z1', d2_col='z2', weights_col='point_weight')
		p = initial_p_hat

		return mu, sigma, p, data


	def _sweep_covariance_initialization(self, observed_df, gene_1, gene_2, group, search_num):
		""" Get an initialization point for 2D EM by sweeping parameters. """

		mu_1 = self.param_1d[gene_1][group][0]
		mu_2 = self.param_1d[gene_2][group][0]
		mu = np.array([mu_1, mu_2])
		sigma_1 = self.param_1d[gene_1][group][1]
		sigma_2 = self.param_1d[gene_2][group][1]
		p = self.param_1d[gene_2][group][2]

		max_magnitude = sigma_1 * sigma_2

		search_parameters = np.linspace(-max_magnitude+0.01, max_magnitude-0.01, search_num)
		ll_estimates = []
		for idx, cov_val in enumerate(search_parameters):

			cov_estimate = np.array([[sigma_1**2, cov_val], [cov_val, sigma_2**2]])

			px_table = self._create_px_table_2d(mu, cov_estimate, p, 'lognormal')

			observed_df['ll'] = observed_df\
				.apply(
					lambda row: row['count'] * np.log(px_table[int(row[gene_1]), int(row[gene_2])]) if max(row[gene_1], row[gene_2]) < self.max_latent_2d else 0, axis=1)
			ll_estimates.append(observed_df['ll'].sum())
		ll_estimates = np.array(ll_estimates)

		cov_estimate = search_parameters[ll_estimates == ll_estimates.max()]

		if gene_1 + '*' + gene_2 not in self.param_sweep_2d:
			self.param_sweep_2d[gene_1 + '*' + gene_2] = {}
		self.param_sweep_2d[gene_1 + '*' + gene_2][group] = search_parameters.copy()

		if gene_1 + '*' + gene_2 not in self.log_likelihood_2d:
			self.log_likelihood_2d[gene_1 + '*' + gene_2] = {}
		self.log_likelihood_2d[gene_1 + '*' + gene_2][group] = ll_estimates.copy()
		if len(cov_estimate) > 1:

			return None, None

		else:

			return mu, np.array([[sigma_1**2, cov_estimate[0]], [cov_estimate[0], sigma_2**2]])


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

		# Format the data
		data = pd.DataFrame()
		data['observed'] = observed
		data = data.groupby('observed').size().reset_index(name='count')
		data['observed_weight'] = data['count'] / len(observed)
		data.set_index('observed', inplace=True)

		# Do a QC check for zero inflation (negative mean)
		neg_mean = (observed_counts[0]/observed_counts.sum() > 0.9)

		# Fitting progress
		fitting_progress = []
		for itr in range(num_iter):

			# Compute the log likelihood
			px_table = self._create_px_table(mu_hat, sigma_hat, p_hat)
			log_likelihood = np.array([count*np.log(px_table[int(val)]) if val < self.max_latent else 0 for val,count in zip(observed_counts.index, observed_counts)]).sum()
			

			# Early stopping and prevent pathological behavior
			stopping_condition = \
				itr > 0 and \
				(np.isinf(log_likelihood) or \
					np.isnan(log_likelihood) or \
					(log_likelihood < fitting_progress[-1][4] + 1e-3) or \
					np.isnan(mu_hat) or \
					np.isnan(sigma_hat))

			if stopping_condition:
				mu_hat = fitting_progress[-1][1]
				sigma_hat = fitting_progress[-1][2]
				print("stopped", itr)
				break

			# Keep track of the fitting progress
			fitting_progress.append((itr, mu_hat, sigma_hat, p_hat, log_likelihood))

			# E step
			prob_table = self._create_pz_table(mu_hat, sigma_hat, p_hat, px_table)

			# M step
			mu_hat, sigma_hat, p_hat = self._get_parameters(data, prob_table, p_hat, neg_mean=neg_mean)

		fitting_progress = pd.DataFrame(fitting_progress, columns=['iteration', 'mu_hat', 'sigma_hat', 'p_hat', 'log_likelihood'])

		if gene not in self.param_1d:
			self.param_1d[gene] = {}
			self.fitting_progress[gene] = {}
		self.param_1d[gene][group] = (mu_hat, sigma_hat, p_hat, observed.shape[0])
		self.fitting_progress[gene][group] = fitting_progress.copy()


	def compute_2d_params(
		self,
		gene_1, 
		gene_2, 
		group='all',
		search_num=30,
		num_iter=30,
		):

		if gene_1 > gene_2: # Flip for convention

			temp = gene_1
			gene_1 = gene_2
			gene_2 = temp

		if not (gene_1 in self.param_1d and group in self.param_1d[gene_1]):

			print('Please run 1D parameter estimation first!')
			raise

		if not (gene_2 in self.param_1d and group in self.param_1d[gene_2]):

			print('Please run 1D parameter estimation first!')
			raise

		if gene_1 + '*' + gene_2 in self.param_2d and group in self.param_2d[gene_1 + '*' + gene_2]:

			print('Already computed!')
			return


		cell_selector = (self.anndata.obs[self.group_label] == group).values if group != 'all' else np.arange(self.anndata.shape[0])
		gene_selector = (self.anndata.var.index.values == gene_1) + (self.anndata.var.index.values == gene_2)
		observed = self.anndata.X[cell_selector, :][:, gene_selector]

		if type(self.anndata.X) != np.ndarray:
			observed = observed.toarray().reshape(-1, 2)

		observed_df = \
			pd.DataFrame(observed, columns=[gene_1, gene_2])\
				.groupby([gene_1, gene_2])\
				.size()\
				.reset_index(name='count')
		observed_df['observed_weight'] = observed_df['count'] / len(observed)

		mu_initial, cov_initial = self._sweep_covariance_initialization(
			observed_df, 
			gene_1, 
			gene_2, 
			group,
			search_num)

		print('Search done')

		observed_df.drop('ll', axis=1, inplace=True)

		print(mu_initial)
		print(cov_initial)
		# TODO: Implement 2D EM procedure

		mu_hat = mu_initial.copy()
		cov_hat = cov_initial.copy()
		p_hat = self.p

		neg_mean = False

		# Fitting progress
		fitting_progress = []
		for itr in range(num_iter):

			# Compute the log likelihood
			px_table = self._create_px_table_2d(mu_hat, cov_hat, self.p)
			
			log_likelihood = np.array([count*np.log(px_table[int(d1), int(d2)]) if max(d1, d2) < self.max_latent_2d else 0 for d1, d2,count in zip(observed_df[gene_1], observed_df[gene_1], observed_df['count'])]).sum()
			

			# Early stopping and prevent pathological behavior
			stopping_condition = \
				itr > 0 and \
				(np.isinf(log_likelihood) or \
					np.isnan(log_likelihood) or \
					(log_likelihood < fitting_progress[-1][4] + 1e-3))
			fitting_progress.append((itr, mu_hat, cov_hat, p_hat, log_likelihood))
			

			if stopping_condition:
				print('Stopped')
				# mu_hat = fitting_progress[-1][1]
				# cov_hat = fitting_progress[-1][2]
				break

			# Keep track of the fitting progress
			#fitting_progress.append((itr, mu_hat, cov_hat, p_hat, log_likelihood))

			# E step
			prob_table = self._create_pz_table_2d(mu_hat, cov_hat, self.p, px_table).query('latent_weight > 0')

			# M step
			mu_hat, cov_hat, p_hat, data = self._get_parameters_2d(
				observed_df.rename(columns={gene_1:'x1', gene_2:'x2'}), 
				prob_table, p_hat, neg_mean=neg_mean)

			return data

		fitting_progress = pd.DataFrame(fitting_progress, columns=['iteration', 'mu_hat', 'cov_hat', 'p_hat', 'log_likelihood'])

		if gene_1 + '*' + gene_2 not in self.param_2d:
			self.param_2d[gene_1 + '*' + gene_2] = {}	
			self.fitting_progress_2d[gene_1 + '*' + gene_2] = {}	
		self.param_2d[gene_1 + '*' + gene_2][group] = (mu_hat, cov_hat)
		self.fitting_progress_2d[gene_1 + '*' + gene_2][group] = fitting_progress.copy()

		return observed_df, px_table


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
			print('Please fit the parameters first!')
			raise

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


	def compute_correlation(self, gene_1, gene_2, group):
		"""
			Compute the correlation from the covariance and variance measurements.
		"""

		corr = self.param_2d[gene_1 + '*' + gene_2][group] / (self.param_1d[gene_1][group][1]*self.param_1d[gene_2][group][1])

		return corr


	def export_model(self, save_path):
		"""
			Collect all of the properties into a dictionary and pickle it.
		"""

		model_info = {}
		model_info['p'] = self.p
		model_info['group_label'] = self.group_label
		model_info['param_1d'] = self.param_1d
		model_info['param_2d'] = self.param_2d
		model_info['param_sweep_2d'] = self.param_sweep_2d
		model_info['fitting_progress'] = self.fitting_progress
		model_info['diff_exp'] = self.diff_exp

		with open(save_path, 'wb') as f:
			pkl.dump(model_info, f)


	def import_model(self, model_path):
		"""
			Take a model export and re-populate the fields
		"""

		with open(model_path, 'rb') as f:
			model_info = pkl.load(f)

		self.p = model_info['p']
		self.group_label = model_info['group_label']
		self.param_1d = model_info['param_1d']
		self.param_2d = model_info['param_2d']
		self.param_sweep_2d = model_info['param_sweep_2d']
		self.fitting_progress = model_info['fitting_progress']
		self.diff_exp = model_info['diff_exp']


