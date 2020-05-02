"""
	estimator.py

	Single Cell Method of Moments

	This file contains code for implementing the method of moments estimators for single cell data.
"""


import pandas as pd
import scipy.stats as stats
import scipy.sparse as sparse
import numpy as np
import time
import scipy as sp
from scipy.stats import multivariate_normal
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
import sklearn as sk

from utils import *
from sparse_array import *


class SingleCellEstimator(object):
	"""
		SingleCellEstimator is the class for fitting univariate and bivariate single cell data. 

		:q: Expected value of the capture rate.
	"""


	def __init__(
		self, 
		adata,
		covariate_label=None,
		replicate_label=None,
		batch_label=None,
		subsection_label=None,
		num_permute=10000,
		label_delimiter='^',
		covariate_converter={},
		q=0.1,
		smooth=True,
		use_hat_matrix=False):

		# Copy over the anndata object
		self.anndata = adata.copy()
		self.is_dense = type(adata.X) != np.ndarray

		# Keep q and labels
		self.q = q
		self.group_label = 'scmemo_group'
		self.covariate_label = covariate_label if covariate_label else 'default_cov'
		self.replicate_label = replicate_label if replicate_label else 'default_rep'
		self.batch_label = batch_label if batch_label else 'default_batch'
		self.subsection_label = subsection_label
		self.label_delimiter = label_delimiter

		# Form discrete groups
		if covariate_label is None:
			self.anndata.obs[self.covariate_label] = 'default_cov'
		if replicate_label is None:
			self.anndata.obs[self.replicate_label] = 'default_rep'
		if batch_label is None:
			self.anndata.obs[self.batch_label] = 'default_batch'
		if subsection_label is None:
			self.anndata.obs[self.subsection_label] = 'default_subsection'

		self.anndata.obs[self.group_label] = 'sg' + self.label_delimiter + \
			self.anndata.obs[self.covariate_label].astype(str) + self.label_delimiter + \
			self.anndata.obs[self.replicate_label].astype(str) + self.label_delimiter + \
			self.anndata.obs[self.batch_label].astype(str) + self.label_delimiter + \
			self.anndata.obs[self.subsection_label].astype(str)
		self.groups = self.anndata.obs[self.group_label].drop_duplicates().tolist() + ['all']
		
		# Keep n_umis, num_permute, cov converter
		self.n_umis = adata.X.sum(axis=1).A1
		self.num_permute = num_permute
		self.covariate_converter = covariate_converter
		self.smooth = smooth
		
		# Initialize mean-var relationship params
		self.mean_var_slope = None
		self.mean_var_inter = None

		# Initialize parameter containing dictionaries
		self.observed_moments = {}
		self.observed_central_moments = {}
		self.estimated_central_moments = {}
		self.parameters = {}
		self.parameters_confidence_intervals = {}

		# Attributes for hypothesis testing
		self.use_hat_matrix = use_hat_matrix
		self.hypothesis_test_result = {}

		# Cache for selecting cells
		self.group_cells = {}
		
		# Initialize the dictionarys
		self._init_dicts()
	
	
	def _init_dicts(self):
		""" Fill the parameter and hypothesis testing dicts with empty sparse matrices. """
		
		# Fill the dictionaries containing moments and estimated parameters
		for group in self.groups:
			
			self.observed_moments[group] = {}
			self.observed_moments[group]['first'] = sparse_array()
			self.observed_moments[group]['second'] = sparse_array()
			self.observed_moments[group]['prod'] = sparse.lil_matrix((self.anndata.shape[1], self.anndata.shape[1]), dtype=np.float32)
			
			self.observed_central_moments[group] = {}
			self.observed_central_moments[group]['first'] = sparse_array()
			self.observed_central_moments[group]['second'] = sparse_array()
			self.observed_central_moments[group]['prod'] = sparse.lil_matrix((self.anndata.shape[1], self.anndata.shape[1]), dtype=np.float32)

			self.estimated_central_moments[group] = {}
			self.estimated_central_moments[group]['first'] = sparse_array()
			self.estimated_central_moments[group]['second'] = sparse_array()
			self.estimated_central_moments[group]['prod'] = sparse.lil_matrix((self.anndata.shape[1], self.anndata.shape[1]), dtype=np.float32)

			self.parameters[group] = {}
			self.parameters[group]['mean'] = sparse_array()
			self.parameters[group]['log_mean'] = sparse_array()
			self.parameters[group]['residual_var'] = sparse_array()
			self.parameters[group]['log_residual_var'] = sparse_array()
			self.parameters[group]['corr'] = sparse.lil_matrix((self.anndata.shape[1], self.anndata.shape[1]), dtype=np.float32)
			
			attributes = ['mean', 'residual_var', 'log_mean', 'log_residual_var', 'log1p_mean', 'log1p_residual_var']
			self.parameters_confidence_intervals[group] = {}
			for attribute in attributes:
				self.parameters_confidence_intervals[group][attribute] = sparse_array()
			self.parameters_confidence_intervals[group]['corr'] = sparse.lil_matrix((self.anndata.shape[1], self.anndata.shape[1]), dtype=np.float32)

			
	def _get_covariate(self, group):
		return group.split(self.label_delimiter)[1]


	def _get_replicate(self, group):
		return group.split(self.label_delimiter)[2]


	def _get_batch(self, group):
		return group.split(self.label_delimiter)[3]


	def _get_subsection(self, group):
		return group.split(self.label_delimiter)[4]


	def _select_cells(self, group, n_umis=False):
		""" Select the cells. """

		# If rows are already selected
		if group in self.group_cells and not n_umis:
			return self.group_cells[group]
		
		if group == 'all': # Whole dataset
			
			if n_umis:
				return self.n_umis
			else:
				data = self.anndata.X
		
		else: # select the cells
			cell_selector = (self.anndata.obs[self.group_label] == group).values
			
			if n_umis:
				return self.n_umis[cell_selector]
			else:
				data = self.anndata.X[cell_selector, :]

		self.group_cells[group] = data.tocsc() # convert to CSC format for fast column indexing
		return self.group_cells[group]


	def _get_gene_idxs(self, gene_list):
		""" Returns the indices of each gene in the list. """

		return np.array([np.where(self.anndata.var.index == gene)[0][0] for gene in gene_list]) # maybe use np.isin


	def _estimate_residual_variance(self, estimated_mean, estimated_var):
		""" 
			Estimate the residual variance by performing linear regression. 

			Returns the residual variance as well as its logarithm.
		"""

		log_residual_variance = np.log(estimated_var) - (self.mean_var_slope*np.log(estimated_mean) + self.mean_var_inter)

		return np.exp(log_residual_variance), log_residual_variance


	def _compute_params(self, group, gene_idxs):
		""" 
			Use the estimated moments to compute the parameters of marginal distributions as 
			well as the estimated correlation. 
		"""

		residual_variance, log_residual_variance = self._estimate_residual_variance(
			self.estimated_central_moments[group]['first'][gene_idxs],
			self.estimated_central_moments[group]['second'][gene_idxs])
		
		self.parameters[group]['mean'] = self.estimated_central_moments[group]['first']
		self.parameters[group]['log_mean'][gene_idxs] = np.log(self.estimated_central_moments[group]['first'][gene_idxs])
		self.parameters[group]['residual_var'][gene_idxs] = residual_variance
		self.parameters[group]['log_residual_var'][gene_idxs] = log_residual_variance


	def _compute_estimated_1d_moments(self, group, gene_idxs):
		""" Use the observed moments to compute the moments of the underlying distribution. """

		mean_inv_numis = _compute_mean_inv_numis(
			self.observed_central_moments[group]['allgenes_first'], 
			self.observed_central_moments[group]['allgenes_second'],
			self.q,
			self.q_sq)

		estimated_mean = _estimate_mean(self.observed_moments[group]['first'][gene_idxs], self.q)

		estimated_var = _estimate_variance(
			self.observed_moments[group]['first'][gene_idxs], 
			self.observed_moments[group]['second'][gene_idxs],
			mean_inv_numis,
			self.q,
			self.q_sq)
		estimated_var[estimated_var < 0] = np.nan
		
		self.estimated_central_moments[group]['first'][gene_idxs] = estimated_mean
		self.estimated_central_moments[group]['second'][gene_idxs] = estimated_var


	def _get_effect_size(self, response_variable, test_dict, nan_thresh=0.5):
		""" Perform the regression for a particular subsection. """

		percent_missing, response_variable = _mean_substitution(response_variable)

		if self.use_hat_matrix:
			effect_sizes = test_dict['hat_matrix'].dot(response_variable)[0, :]
		else:
			effect_sizes = \
				sk.linear_model.LinearRegression(fit_intercept=False)\
				.fit(test_dict['design_matrix'], response_variable, test_dict['cell_count']).coef_[:, 0]

		effect_sizes[percent_missing > nan_thresh] = np.nan

		return effect_sizes


	def compute_observed_moments(self, gene_list, verbose=False):
		""" Compute the observed statistics. Does not compute the covariance. """
		
		gene_idxs = self._get_gene_idxs(gene_list)

		for group in self.groups:

			if verbose:
				print('Computing observed moments for:', group)

			observed = self._select_cells(group)[:, gene_idxs]

			first, second = _compute_1d_statistics(observed, smooth=self.smooth)
			allgenes_first = self._select_cells(group, n_umis=True).mean()
			allgenes_second = (self._select_cells(group, n_umis=True)**2).mean()
	
			# Observed moments
			self.observed_moments[group]['first'][gene_idxs] = first
			self.observed_moments[group]['second'][gene_idxs] = second
			self.observed_moments[group]['allgenes_first'] = allgenes_first
			self.observed_moments[group]['allgenes_second'] = allgenes_second

			# Observed central moments
			self.observed_central_moments[group]['first'][gene_idxs] = first
			self.observed_central_moments[group]['second'][gene_idxs] = second-first**2
			self.observed_central_moments[group]['allgenes_first'] = allgenes_first
			self.observed_central_moments[group]['allgenes_second'] = allgenes_second - allgenes_first**2
			

	def estimate_q_sq(self, k=5, verbose=True):
		""" 
			Estimate the expected value of the square of the capture rate q. 

			Estimate the relationship between mean and variance. This means that we assume the mean variance relationship to be 
			the same for all cell types. 
		"""
		
		# Compute observed statistics from all cells
		observed = self._select_cells('all')
		first, second = _compute_1d_statistics(observed, smooth=self.smooth)
		
		# Grab the values needed for fitting q_sq
		x = first
		y = second-first**2

		# Filter for finite estimates
		finite_filter = np.isfinite(x) & np.isfinite(y)
		
		# Filter for genes with max value greater than k
		max_val_filter = (self.anndata.X.max(axis=0).toarray()[0] >= k)
		
		# Filter the genes used to compute q_sq
		x = x[finite_filter & max_val_filter]
		y = y[finite_filter & max_val_filter]
		
		# Save the mean, cv
		self.q_sq_x = x
		self.q_sq_y = y

		# Get the upper and lower limits for q_sq
		lower_lim = self.q**2
		upper_lim = (self.n_umis**2).mean()/self.n_umis.mean()**2*self.q**2


		# Estimate the noise level, or CV^2_q + 1
		noise_level = np.nanmin((y/x**2 - 1/x)[y > x])

		# Estimate the initial guess for q_sq
		initial_q_sq_estimate = (noise_level+1)*self.q**2

		# Refine the initial guess
# 		res = sp.optimize.minimize_scalar(
# 			estimated_mean_disp_corr, 
# 			bounds=[lower_lim, upper_lim], 
# 			args=(self, frac),
# 			method='bounded',
# 			options={'maxiter':100, 'disp':verbose})
		q_sq_estimate = initial_q_sq_estimate
	
		# Bound the estimate
		if q_sq_estimate > upper_lim:
			q_sq_estimate = upper_lim
		if q_sq_estimate < lower_lim:
			q_sq_estimate = lower_lim + 1e-7

		# Clear out estimated parameters
# 		self.parameters = {}

		# Keep estimated parameters for later
		self.noise_level = q_sq_estimate/self.q**2-1
		self.q_sq = q_sq_estimate

		if verbose:
			print('E[q^2] falls in [{:.5f}, {:.8f}], with the current estimate of {:.8f}. Estimated with {} genes.'\
					  .format(lower_lim, upper_lim, self.q_sq, x.shape[0]))


	def plot_cv_mean_curve(self, group='all', estimated=False, plot_noise=True):
		"""
			Plot the observed characteristic relationship between the mean and the coefficient of variation. 

			If an estimate for q_sq exists, also plot the estimated baseline noise level.
		"""

		obs_mean = self.q_sq_x
		obs_var = self.q_sq_y

		plt.scatter(
		    np.log(obs_mean),
		    np.log(obs_var)/2-np.log(obs_mean),
		    s=2,
		    alpha=0.5
		)

		bound_x = np.arange(
		    np.nanmin(obs_mean),
		    np.nanmax(obs_mean),
		    0.01)
		bound_y = 1/bound_x + self.q_sq/self.q**2-1

		plt.plot(np.log(bound_x), -np.log(bound_x)/2, color='k', lw=2)

		if not estimated and plot_noise:
			plt.plot(np.log(bound_x), np.log(bound_y)/2, lw=2, color='r')
		plt.axis('equal');
		plt.legend(['Poisson', 'Poisson + noise', 'genes'])
		plt.title('Observed Mean - CV Relationship');
		plt.xlabel('log( observed mean )')
		plt.ylabel('log( observed CV )')


	def estimate_1d_parameters(self, gene_list):
		""" Perform 1D (mean, variability) parameter estimation. """
		
		gene_idxs = self._get_gene_idxs(gene_list)

		# Compute estimated moments
		for group in self.groups:
			self._compute_estimated_1d_moments(group, gene_idxs)

		# Combine all estimated moments from every group
		x = np.concatenate([
			self.estimated_central_moments[group]['first'][gene_idxs]for group in self.groups])
		y = np.concatenate([
			self.estimated_central_moments[group]['second'][gene_idxs] for group in self.groups])

		# Filter for finite estimates
		condition = np.isfinite(x) & np.isfinite(y)
		x = x[condition]
		y = y[condition]

		# Estimate the mean-var relationship
		if not self.mean_var_slope and not self.mean_var_inter:
			slope, inter, _, _, _ = _robust_linregress(np.log(x), np.log(y))
			self.mean_var_slope = slope
			self.mean_var_inter = inter

		# Compute final parameters
		for group in self.groups:
			self._compute_params(group, gene_idxs)


	def estimate_2d_parameters(self, gene_list_1, gene_list_2, groups='all'):
		""" Perform 2D parameter estimation. """

		groups_to_iter = groups if groups != 'all' else self.groups
		gene_idxs_1 = self._get_gene_idxs(gene_list_1)
		gene_idxs_2 = self._get_gene_idxs(gene_list_2)

		for group in groups_to_iter:

			mean_inv_numis = _compute_mean_inv_numis(
				self.observed_central_moments[group]['allgenes_first'], 
				self.observed_central_moments[group]['allgenes_second'],
				self.q,
				self.q_sq)

			observed = self._select_cells(group)
			observed_1 = observed[:, gene_idxs_1]
			observed_2 = observed[:, gene_idxs_2]

			# Compute the observed cross-covariance and expectation of the product
			observed_cov, observed_prod = _sparse_cross_covariance(observed_1, observed_2)

			# Update the observed dictionaries
			self.observed_moments[group]['prod'][gene_idxs_1[:, np.newaxis], gene_idxs_2] = observed_prod
			self.observed_central_moments[group]['prod'][gene_idxs_1[:, np.newaxis], gene_idxs_2] = observed_cov

			# Estimate the true covariance
			estimated_cov = _estimate_covariance(
				self.observed_central_moments[group]['first'][gene_idxs_1],
				self.observed_central_moments[group]['first'][gene_idxs_2],
				observed_prod,
				mean_inv_numis,
				self.q,
				self.q_sq)

			# Estimate the true correlation
			vars_1 = self.estimated_central_moments[group]['second'][gene_idxs_1]
			vars_2 = self.estimated_central_moments[group]['second'][gene_idxs_2]
			estimated_corr = estimated_cov / np.sqrt(vars_1[:, np.newaxis]).dot(np.sqrt(vars_2[np.newaxis, :]))
			estimated_corr[estimated_corr > 1] = 1
			estimated_corr[estimated_corr < -1] = -1

			# Update the estimated dictionaries
			self.estimated_central_moments[group]['prod'][gene_idxs_1[:, np.newaxis], gene_idxs_2] = estimated_cov
			self.parameters[group]['corr'][gene_idxs_1[:, np.newaxis], gene_idxs_2] = estimated_corr


	def setup_hypothesis_testing(self, subsections=[]):
		"""
			Perform operations necessarily to set up hypothesis testing for both 1D and 2D. 

			Construct the design matrix. If applicable, compute the hat matrix for each subsection.
		"""

		attributes_1d = [
			'de_effect_size', 'de_pval', 'de_fdr', 'de_es_ci', 
			'dv_effect_size', 'dv_pval', 'dv_fdr', 'dv_es_ci']
		attributes_2d = ['dc_effect_size', 'dc_pval', 'dc_fdr', 'dc_es_ci']


		for subsection in subsections:

			# Setup the hypothesis result dict
			self.hypothesis_test_result[subsection] = {}

			# List of meta observations to construct DF later
			covariate_list = []

			# Get all the appropriate groups
			subsection_groups = [group for group in self.groups if group != 'all' and self._get_subsection(group) == subsection]
			self.hypothesis_test_result[subsection]['groups'] = subsection_groups

			# Iterate through each group
			for group in subsection_groups:

				# Ignore the groups in other subsections
				if self._get_subsection(group) != subsection:
					continue

				covariate, replicate, batch = self._get_covariate(group), self._get_replicate(group), self._get_batch(group)
				covariate_list.append((
					self.covariate_converter[covariate], # Numerical covariate
					replicate, # Replicate (e.g., individual)
					batch, # Batch
					1, # Constant for fitting the intercept
					self.group_cells[group].shape[0], # Number of cells in this group for weighting
					))
			
			subsection_design_df = pd.DataFrame(covariate_list, columns=['covariate', 'replicate', 'batch', 'constant', 'cell_count'])
			self.hypothesis_test_result[subsection]['design_df'] = subsection_design_df.copy()

			# Save the cell counts
			cell_count = subsection_design_df['cell_count'].values
			self.hypothesis_test_result[subsection]['cell_count'] = cell_count

			# Construct the design matrix
			if subsection_design_df['batch'].nunique() < 2:
				design_matrix = subsection_design_df[['covariate', 'constant']].values
			else:
				design_matrix = pd.get_dummies(subsection_design_df[['covariate', 'batch', 'constant']], columns=['batch'], drop_first=True).values
			self.hypothesis_test_result[subsection]['design_matrix'] = design_matrix

			# Compute the hat matrix if applicable
			if self.use_hat_matrix:
				hat_matrix = np.linalg.inv(design_matrix.T.dot(cell_count.reshape(-1,1)*design_matrix)).dot(design_matrix.T*cell_count)
				self.hypothesis_test_result[subsection]['hat_matrix'] = hat_matrix

			# Create empty data structures for hypothesis test results
			for attribute in attributes_1d:
				self.hypothesis_test_result[subsection][attribute] = np.full(self.anndata.shape[1], np.nan)
			for attribute in attributes_2d:
				self.hypothesis_test_result[subsection][attribute] = sparse.lil_matrix(
				(self.anndata.shape[1], self.anndata.shape[1]), dtype=np.float32)


	def compute_effect_sizes_1d(self, gene_list):
		""" 
			Compute the effect sizes for mean and variability. 
			This function does not compute p-values or FDR.
			Assumes that the 1D parameters have been estimated.
			Assumes that the setup_hypothesis_testing function has been run already.
		"""
		
		gene_idxs = self._get_gene_idxs(gene_list)

		for subsection, test_dict in self.hypothesis_test_result.items():

			log_means = np.vstack([self.parameters[group]['log_mean'][gene_idxs] for group in test_dict['groups']])
			log_residual_vars = np.vstack([self.parameters[group]['log_residual_var'][gene_idxs] for group in test_dict['groups']])

			mean_es = self._get_effect_size(log_means, test_dict)
			var_es = self._get_effect_size(log_residual_vars, test_dict)
			
			print(mean_es.shape, var_es.shape)

			test_dict['de_effect_size'][gene_idxs], test_dict['dv_effect_size'][gene_idxs] = mean_es, var_es


	def compute_effect_sizes_2d(self, gene_list_1, gene_list_2):
		"""
			Compute the effect sizes for correlations.
			This function does not compute p-values or FDR.
			Assumes that the 2D parameters for the same gene_lists have been estimated. 
			Assumes that the setup_hypothesis_testing function has been run already.
		"""

		gene_idxs_1 = self._get_gene_idxs(gene_list_1)
		gene_idxs_2 = self._get_gene_idxs(gene_list_2)

		for subsection, test_dict in self.hypothesis_test_result.items():

			correlations = np.vstack([self.parameters[group]['corr'][gene_idxs_1, :][:, gene_idxs_2].toarray().ravel()
					for group in test_dict['groups']])

			corr_es = self._get_effect_size(correlations, test_dict)

			test_dict['dc_effect_size'][gene_idxs_1[:, np.newaxis], gene_idxs_2] = \
				corr_es.reshape(gene_idxs_1.shape[0], gene_idxs_2.shape[0])


	def compute_confidence_intervals_1d(self, gene_list, hypothesis_test=True, gene_tracker_count=100, verbose=False, timer='off'):
		"""
			Compute confidence intervals and p-values for estimate and effect sizes. 

			Use the multinomial resampling.

			Uses self.num_permute attribute, same as in hypothesis testing.

			CAVEAT: Uses the same expectation of 1/N as the true value, does not compute this from the permutations. 
			So the result might be slightly off.
		"""
		
		gene_idxs = self._get_gene_idxs(gene_list)

		# Get the starting time
		start_time = time.time()

		# Calculate the pseudocount, so that the demonimator is N+1
		pseudocount = 1/self.anndata.shape[1] if self.smooth else 0
		
		# Get the relevant groups to iterate over. 
		if hypothesis_test:
			groups_to_iter = []
			for subsection, test_dict in self.hypothesis_test_result.items():
				groups_to_iter += test_dict['groups']
		else: 
			groups_to_iter = self.groups

		# Compute the inverse of N
		mean_inv_numis = {
			group:_compute_mean_inv_numis(
				self.observed_central_moments[group]['allgenes_first'], 
				self.observed_central_moments[group]['allgenes_second'],
				self.q,
				self.q_sq) for group in groups_to_iter}

		# Iterate through each gene and compute a standard error for each gene
		iter = 0
		for gene_idx in gene_idxs:
			
			if verbose and gene_tracker_count > 0 and iter % gene_tracker_count == 0: 
				print('Computing the {}st/th gene, {:.5f} seconds have passed.'.format(iter, time.time()-start_time))

			gene_mult_rvs = {}
			gene_counts = {}
			gene_freqs = {}
			for group in groups_to_iter:

				# Grab the values
				data = self._select_cells(group)
				
				# Get a frequency count
				count_start_time = time.time()
				counts = _sparse_bincount(data[:, gene_idx])
				count_time = time.time() - count_start_time
				
				expr_values = np.arange(counts.shape[0])
				expr_values = expr_values[counts != 0]
				counts = counts[counts != 0]
				gene_counts[group] = expr_values
				gene_freqs[group] = counts/data.shape[0]
			
			compute_start_time = time.time()
			
			for group in groups_to_iter:
				gene_mult_rvs[group] = stats.multinomial.rvs(n=self.group_cells[group].shape[0], p=gene_freqs[group], size=self.num_permute)

			# Construct the repeated values matrix
			values = {group:np.tile(
				gene_counts[group].reshape(1, -1), (self.num_permute, 1)) for group in groups_to_iter}

			# Compute the permuted, observed mean/dispersion
			mean = {group:((gene_mult_rvs[group] * values[group]).sum(axis=1)+pseudocount)/(self.group_cells[group].shape[0]+1) for group in groups_to_iter}
			second_moments = {group:((gene_mult_rvs[group] * values[group]**2).sum(axis=1)+pseudocount)/(self.group_cells[group].shape[0]+1) for group in groups_to_iter}
			del gene_mult_rvs, gene_counts

			# Compute the permuted, estimated moments for both groups
			estimated_means = {group:_estimate_mean(mean[group], self.q) for group in groups_to_iter}
			estimated_vars = {group:_estimate_variance(mean[group], second_moments[group], mean_inv_numis[group], self.q, self.q_sq) for group in groups_to_iter}
			estimated_residual_vars = {group:self._estimate_residual_variance(estimated_means[group], estimated_vars[group])[0] for group in groups_to_iter}
			del mean, second_moments

			compute_time = time.time()-compute_start_time
			
			# Store the S.E. of the parameter, log(param), and log1p(param)		
			for group in groups_to_iter:

				self.parameters_confidence_intervals[group]['mean'][gene_idx] = np.nanstd(estimated_means[group])
				self.parameters_confidence_intervals[group]['residual_var'][gene_idx] = np.nanstd(estimated_residual_vars[group])
				self.parameters_confidence_intervals[group]['log_mean'][gene_idx] = np.nanstd(np.log(estimated_means[group]))
				self.parameters_confidence_intervals[group]['log_residual_var'][gene_idx] = np.nanstd(np.log(estimated_residual_vars[group]))
				self.parameters_confidence_intervals[group]['log1p_mean'][gene_idx] = np.nanstd(np.log(estimated_means[group]+1))
				self.parameters_confidence_intervals[group]['log1p_residual_var'][gene_idx] = np.nanstd(np.log(estimated_residual_vars[group]+1))
			
			# Perform hypothesis testing
			for subsection, test_dict in self.hypothesis_test_result.items():

				# Organize the data into format for meta-analysis
				boot_log_means = np.vstack([np.log(estimated_means[group]) for group in test_dict['groups']])
				boot_log_residual_vars = np.vstack([np.log(estimated_residual_vars[group]) for group in test_dict['groups']])

				# Compute the effect sizes
				mean_es = self._get_effect_size(boot_log_means, test_dict)
				var_es = self._get_effect_size(boot_log_residual_vars, test_dict)

				# Update the test dict
				if np.isfinite(self.hypothesis_test_result[subsection]['de_effect_size'][gene_idx]):
					test_dict['de_es_ci'][gene_idx] = np.nanstd(mean_es)
					test_dict['de_pval'][gene_idx] = _compute_asl(mean_es)
				if np.isfinite(self.hypothesis_test_result[subsection]['dv_effect_size'][gene_idx]):
					test_dict['dv_es_ci'][gene_idx] = np.nanstd(var_es)
					test_dict['dv_pval'][gene_idx] = _compute_asl(var_es)
			iter += 1

		# Perform FDR correction
		for subsection, test_dict in self.hypothesis_test_result.items():
			test_dict['de_fdr'][gene_idxs] = _fdrcorrect(test_dict['de_pval'][gene_idxs])
			test_dict['dv_fdr'][gene_idxs] = _fdrcorrect(test_dict['dv_pval'][gene_idxs])

		if timer == 'on':
			return count_time, compute_time, sum([values[group].shape[1] for group in groups_to_iter])/len(groups_to_iter)
		

	def compute_confidence_intervals_2d(self, gene_list_1, gene_list_2, hypothesis_test=True, gene_tracker_count=100, verbose=False):
		"""
			Compute confidence intervals and p-values for estimate and effect sizes. 
			Use the multinomial resampling.
			Uses self.num_permute attribute, same as in hypothesis testing.
			CAVEAT: Uses the same expectation of 1/N as the true value, does not compute this from the permutations. 
			So the result might be slightly off.
		"""

		# Calculate the pseudocount, so that the demonimator is N+1
		pseudocount = 1/self.anndata.shape[1] if self.smooth else 0
		
		# Get the relevant groups to iterate over. 
		if hypothesis_test:
			groups_to_iter = []
			for subsection, test_dict in self.hypothesis_test_result.items():
				groups_to_iter += test_dict['groups']
		else: 
			groups_to_iter = self.groups

		# Compute the inverse of N
		mean_inv_numis = {
			group:_compute_mean_inv_numis(
				self.observed_central_moments[group]['allgenes_first'], 
				self.observed_central_moments[group]['allgenes_second'],
				self.q,
				self.q_sq) for group in groups_to_iter}

		# Get the gene idx for the two gene lists
		genes_idxs_1 = self._get_gene_idxs(gene_list_1)
		genes_idxs_2 = self._get_gene_idxs(gene_list_2)

		# Iterate through each gene and compute a standard error for each gene
		iter_1 = 0
		iter_2 = 0
		for gene_idx_1 in genes_idxs_1:

			iter_2 = 0
			for gene_idx_2 in genes_idxs_2:

				if gene_idx_2 == gene_idx_1:
					continue

				if verbose and gene_tracker_count > 0 and (iter_1*genes_idxs_2.shape[0] + iter_2) % gene_tracker_count == 0: 
					print('Computing the {}st/th gene of {}'.format((iter_1*genes_idxs_2.shape[0] + iter_2), genes_idxs_1.shape[0]*genes_idxs_2.shape[0]))

				gene_mult_rvs = {}
				cantor_codes = {}
				for group in groups_to_iter:

					# Grab the values
					data = self._select_cells(group)

					cantor_code = _pair(
						data[:, gene_idx_1].toarray().reshape(-1), 
						data[:, gene_idx_2].toarray().reshape(-1))
					expr_values, counts = np.unique(cantor_code, return_counts=True)
					cantor_codes[group] = expr_values
					gene_mult_rvs[group] = stats.multinomial.rvs(n=data.shape[0], p=counts/data.shape[0], size=self.num_permute)
				
				# Construct the repeated values matrix
				cantor_code = {group:cantor_codes[group] for group in groups_to_iter}
				values_1 = {}
				values_2 = {}

				for group in groups_to_iter:
					values_1_raw, values_2_raw = _depair(cantor_code[group])
					values_1[group] = np.tile(values_1_raw.reshape(1, -1), (self.num_permute, 1))
					values_2[group] = np.tile(values_2_raw.reshape(1, -1), (self.num_permute, 1))

				# Compute the bootstrapped observed moments
				mean_1 = {group:((gene_mult_rvs[group] * values_1[group]).sum(axis=1)+pseudocount)/(self.group_cells[group].shape[0]+1) for group in groups_to_iter}
				second_moments_1 = {group:((gene_mult_rvs[group] * values_1[group]**2).sum(axis=1)+pseudocount)/(self.group_cells[group].shape[0]+1) for group in groups_to_iter}
				mean_2 = {group:((gene_mult_rvs[group] * values_2[group]).sum(axis=1)+pseudocount)/(self.group_cells[group].shape[0]+1) for group in groups_to_iter}
				second_moments_2 = {group:((gene_mult_rvs[group] * values_2[group]**2).sum(axis=1)+pseudocount)/(self.group_cells[group].shape[0]+1) for group in groups_to_iter}
				prod = {group:((gene_mult_rvs[group] * values_1[group] * values_2[group]).sum(axis=1)+pseudocount)/(self.group_cells[group].shape[0]+1) for group in groups_to_iter}
				del gene_mult_rvs

				# Compute the permuted, estimated moments for both groups
				estimated_means_1 = {group:_estimate_mean(mean_1[group], self.q) for group in groups_to_iter}
				estimated_vars_1 = {group:_estimate_variance(mean_1[group], second_moments_1[group], mean_inv_numis[group], self.q, self.q_sq) for group in groups_to_iter}
				estimated_means_2 = {group:_estimate_mean(mean_2[group], self.q) for group in groups_to_iter}
				estimated_vars_2 = {group:_estimate_variance(mean_2[group], second_moments_2[group], mean_inv_numis[group], self.q, self.q_sq) for group in groups_to_iter}
				del mean_1, mean_2, second_moments_1, second_moments_2

				# Compute estimated correlations
				estimated_corrs = {}
				for group in groups_to_iter:
					denom = self.q_sq - (self.q - self.q_sq)*mean_inv_numis[group]
					cov = prod[group] / denom - (estimated_means_1[group] * estimated_means_2[group])
					estimated_corrs[group] = cov / np.sqrt(estimated_vars_1[group]*estimated_vars_2[group]) 

				# Store the S.E. of the correlation for each group
				for group in groups_to_iter:
					self.parameters_confidence_intervals[group]['corr'][gene_idx_1, gene_idx_2] = np.nanstd(estimated_corrs[group])

				# Perform hypothesis testing
				for subsection, test_dict in self.hypothesis_test_result.items():

					# Compute the dc bootstrapped effect sizes
					correlations = np.vstack([estimated_corrs[group] for group in test_dict['groups']])
					corr_es = self._get_effect_size(correlations, test_dict)

					# Update the test dict
					if np.isfinite(test_dict['dc_effect_size'][gene_idx_1, gene_idx_2]):
						test_dict['dc_pval'][gene_idx_1, gene_idx_2] = _compute_asl(corr_es)
						test_dict['dc_es_ci'][gene_idx_1, gene_idx_2] = np.nanstd(corr_es)

				iter_2 += 1
			iter_1 += 1

		# Perform FDR correction
		for subsection, test_dict in self.hypothesis_test_result.items():

			test_dict['dc_fdr'][genes_idxs_1[:, np.newaxis], genes_idxs_2] = \
				_fdrcorrect(test_dict['dc_pval'][genes_idxs_1[:, np.newaxis], genes_idxs_2].toarray().ravel())\
				.reshape(genes_idxs_1.shape[0], genes_idxs_2.shape[0])
