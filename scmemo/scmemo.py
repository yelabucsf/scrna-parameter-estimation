"""
	scmemo.py
	
	This file contains the public facing API for using scmemo.
"""


import numpy as np
import pandas as pd
from patsy import dmatrix
import scipy.stats as stats
from multiprocessing import Pool
from functools import partial

import bootstrap
import estimator
import hypothesis_test
import util


def create_groups(
	adata, 
	label_columns, 
	label_delimiter='^', 
	inplace=True,
	):
	"""
		Creates discrete groups of the data, based on the columns in :label_columns:
	"""
	
	if not inplace:
		adata = adata.copy()
		
	# Create group labels
	adata.obs['scmemo_group'] = 'sg' + label_delimiter
	for idx, col_name in enumerate(label_columns):
		adata.obs['scmemo_group'] += adata.obs[col_name].astype(str)
		if idx != len(label_columns)-1:
			adata.obs['scmemo_group'] += label_delimiter
	
	# Create a dict in the uns object
	adata.uns['scmemo'] = {}
	adata.uns['scmemo']['label_columns'] = label_columns
	adata.uns['scmemo']['label_delimiter'] = label_delimiter
	adata.uns['scmemo']['groups'] = adata.obs['scmemo_group'].drop_duplicates().tolist()
	
	# Create slices of the data based on the group
	adata.uns['scmemo']['group_cells'] = {group:util._select_cells(adata, group) for group in adata.uns['scmemo']['groups']}
	
	if not inplace:
		return adata

def compute_1d_moments(
	adata, 
	inplace=True, 
	filter_mean_thresh=0.07, 
	min_perc_group=0.7, 
	filter_genes=True, 
	residual_var=True,
	use_n_umi=True):
	
	assert 'scmemo' in adata.uns
	
	if not inplace:
		adata = adata.copy()
		
	# Compute n_umi for the entire dataset
	adata.uns['scmemo']['n_umi'] = adata.X.sum(axis=1).mean() if use_n_umi else 1
		
	# Compute size factors for all groups
	size_factor = estimator._estimate_size_factor(adata.X)
	
# 	# Bin the size factors
# 	binned_stat = stats.binned_statistic(size_factor, size_factor, bins=50, statistic='median')
# 	bin_idx = np.clip(binned_stat[2], a_min=1, a_max=binned_stat[0].shape[0])
# 	approx_sf = binned_stat[0][bin_idx-1]
# 	max_sf = size_factor.max()
# 	approx_sf[size_factor == max_sf] = max_sf
# 	size_factor = approx_sf
	
	adata.uns['scmemo']['all_size_factor'] = size_factor
	adata.uns['scmemo']['size_factor'] = \
		{group:size_factor[(adata.obs['scmemo_group'] == group).values] for group in adata.uns['scmemo']['groups']}
	
	# Compute 1d moments for all groups
	adata.uns['scmemo']['1d_moments'] = {group:estimator._poisson_1d(
		data=adata.uns['scmemo']['group_cells'][group],
		n_obs=adata.uns['scmemo']['group_cells'][group].shape[0],
		size_factor=adata.uns['scmemo']['size_factor'][group],
		n_umi=adata.uns['scmemo']['n_umi']) for group in adata.uns['scmemo']['groups']}
	
	# Create gene masks for each group
	adata.uns['scmemo']['gene_filter'] = \
		{group:(adata.uns['scmemo']['1d_moments'][group][0] > filter_mean_thresh/adata.uns['scmemo']['n_umi'] ) 
		 for group in adata.uns['scmemo']['groups']}
	
	# Create overall gene mask
	gene_masks = np.vstack([adata.uns['scmemo']['gene_filter'][group] for group in adata.uns['scmemo']['groups']])
	gene_filter_rate = gene_masks.mean(axis=0)
	overall_gene_mask = (gene_filter_rate > min_perc_group)
	adata.uns['scmemo']['overall_gene_filter'] = overall_gene_mask
	adata.uns['scmemo']['gene_list'] = adata.var.index[overall_gene_mask].tolist()
	
	# Filter the genes from the data matrices as well as the 1D moments
	if filter_genes:
		adata.uns['scmemo']['group_cells'] = \
			{group:adata.uns['scmemo']['group_cells'][group][:, overall_gene_mask] for group in adata.uns['scmemo']['groups']}
		adata.uns['scmemo']['1d_moments'] = \
			{group:[
				adata.uns['scmemo']['1d_moments'][group][0][overall_gene_mask],
				adata.uns['scmemo']['1d_moments'][group][1][overall_gene_mask]
				] for group in adata.uns['scmemo']['groups']}
		adata._inplace_subset_var(overall_gene_mask)
	
	# Compute residual variance
	if residual_var:
	
		adata.uns['scmemo']['mv_regressor'] = {}
		for group in adata.uns['scmemo']['groups']:
			
			adata.uns['scmemo']['mv_regressor'][group] = estimator._fit_mv_regressor(
				mean=adata.uns['scmemo']['1d_moments'][group][0],
				var=adata.uns['scmemo']['1d_moments'][group][1])
			
			res_var = estimator._residual_variance(
				adata.uns['scmemo']['1d_moments'][group][0],
				adata.uns['scmemo']['1d_moments'][group][1],
				adata.uns['scmemo']['mv_regressor'][group])
			adata.uns['scmemo']['1d_moments'][group].append(res_var)
	
	if not inplace:
		return adata


def compute_2d_moments(adata, gene_1, gene_2, inplace=True):
	"""
		Compute the covariance and correlation for given genes.
		This function computes the covariance and the correlation between genes in :gene_1: and genes in :gene_2:. 
	"""
	
	if not inplace:
		adata = adata.copy()
		
	# Set up the result dictionary
	adata.uns['scmemo']['2d_moments'] = {}
	adata.uns['scmemo']['2d_moments']['gene_1'] = gene_1
	adata.uns['scmemo']['2d_moments']['gene_2'] = gene_2
	adata.uns['scmemo']['2d_moments']['gene_idx_1'] = util._get_gene_idx(adata, gene_1)
	adata.uns['scmemo']['2d_moments']['gene_idx_2'] = util._get_gene_idx(adata, gene_2)
	
	for group in adata.uns['scmemo']['groups']:
		
		cov = estimator._poisson_cov(
			data=adata.uns['scmemo']['group_cells'][group], 
			n_obs=adata.uns['scmemo']['group_cells'][group].shape[0], 
			size_factor=adata.uns['scmemo']['size_factor'][group], 
			idx1=adata.uns['scmemo']['2d_moments']['gene_idx_1'], 
			idx2=adata.uns['scmemo']['2d_moments']['gene_idx_2'],
			n_umi=adata.uns['scmemo']['n_umi'])
		
		var_1 = adata.uns['scmemo']['1d_moments'][group][1][adata.uns['scmemo']['2d_moments']['gene_idx_1']]
		var_2 = adata.uns['scmemo']['1d_moments'][group][1][adata.uns['scmemo']['2d_moments']['gene_idx_2']]
		
		corr = estimator._corr_from_cov(cov, var_1, var_2)
		
		adata.uns['scmemo']['2d_moments'][group] = (cov, corr, var_1, var_2)

	if not inplace:
		return adata

	
def bootstrap_1d_moments(adata, inplace=True, num_boot=10000, verbose=False, bins=5):
	"""
		Computes the CI for mean and variance of each gene 
	"""
	
	if not inplace:
		adata = adata.copy()
		
	adata.uns['scmemo']['1d_ci'] = {}
	for group in adata.uns['scmemo']['groups']:
		
		if verbose:
			print(group)
				
		# Compute 1D CIs
		mean_se, log_mean_se, var_se, log_var_se, res_var_se = bootstrap._bootstrap_1d(
			data=adata.uns['scmemo']['group_cells'][group], 
			size_factor=adata.uns['scmemo']['size_factor'][group], 
			num_boot=num_boot, 
			mv_regressor=adata.uns['scmemo']['mv_regressor'][group],
			n_umi=adata.uns['scmemo']['n_umi'],
			bins=bins)
				
		adata.uns['scmemo']['1d_ci'][group] = [mean_se, log_mean_se, var_se, log_var_se, res_var_se]
		
	if not inplace:
		return adata
		
		
def bootstrap_2d_moments(adata, inplace=True, num_boot=10000):
	"""
		Computes the CI for covariance and correlation of each gene pair 
	"""
	
	if not inplace:
		adata = adata.copy()
		
	adata.uns['scmemo']['2d_ci'] = {}
	for group in adata.uns['scmemo']['groups']:
						
		# Compute 2D CIs
		cov_se, corr_se = bootstrap._bootstrap_2d(
			data=adata.uns['scmemo']['group_cells'][group], 
			size_factor=adata.uns['scmemo']['size_factor'][group],
			gene_idxs_1=adata.uns['scmemo']['2d_moments']['gene_idx_1'],
			gene_idxs_2=adata.uns['scmemo']['2d_moments']['gene_idx_2'],
			num_boot=num_boot,
			n_umi=adata.uns['scmemo']['n_umi'])
		
		adata.uns['scmemo']['2d_ci'][group] = [cov_se, corr_se]
		
	if not inplace:
		return adata

	
def ht_1d_moments(
	adata, 
	formula_like,
	cov_column,
	inplace=True, 
	use_residual_var=True, 
	num_boot=10000, 
	dirichlet_approx=True,
	log=True,
	verbose=False):
	"""
		Performs hypothesis testing for 1D moments.
	"""
	
	if not inplace:
		adata = adata.copy()
	
	# Get number of genes
	G = adata.shape[1]
	
	# Create design DF
	design_df_list, Nc_list = [], []
	
	# Create the design df
	for group in adata.uns['scmemo']['groups']:
		
		design_df_list.append(group.split(adata.uns['scmemo']['label_delimiter'])[1:])
		Nc_list.append(adata.uns['scmemo']['group_cells'][group].shape[0])
		
	# Create the design matrix from the patsy formula
	design_df = pd.DataFrame(design_df_list, columns=adata.uns['scmemo']['label_columns'])
	design_matrix = dmatrix(formula_like, design_df)
	design_matrix_cols = design_matrix.design_info.column_names
	Nc_list = np.array(Nc_list)
	
	# Find the covariate that actually matters
	for idx, col_name in enumerate(design_matrix_cols):
		if cov_column in col_name:
			cov_idx = idx
			break
	
	# Initialize empty arrays to hold bootstrap values
	# First value is always the actual estimate
	boot_mean = np.zeros((design_matrix.shape[0], num_boot+1))*np.nan
	boot_var = np.zeros((design_matrix.shape[0], num_boot+1))*np.nan
	
	# Initialize empty arrays to hold fitted coefficients and achieved significance level
	mean_coef, mean_asl, var_coef, var_asl = [np.zeros(G)*np.nan for i in range(4)]
	
	# Precompute the bootstrap number of cells per group
	# Approximate with Poisson
	weights = stats.poisson.rvs(Nc_list, size=(num_boot+1, Nc_list.shape[0])).T
	weights[:, 0] = Nc_list
	covariate = np.array(design_matrix[:, cov_idx].reshape(-1, 1))
	X = np.tile(covariate, (1, num_boot+1))

	X_mean = np.average(X, axis=0, weights=weights)
	X_centered = X - X_mean
	X_centered_sq = X_centered**2
	w_X_centered = weights*X_centered
	w_X_centered_sq = weights*X_centered_sq
	
	partial_func = partial(
		hypothesis_test._ht_1d,
		adata_dict=adata.uns['scmemo'],
		w_X_centered=w_X_centered,
		w_X_centered_sq=w_X_centered_sq,
		weights=weights,
		use_residual_var=use_residual_var,
		dirichlet_approx=dirichlet_approx,
		log=log)
	
	# Multiprocess
	try:
		pool = Pool(processes=3)
		results = pool.map(partial_func, [idx for idx in range(G)])
		pool.close()
		pool.join()
		
	except:
		pool.close()
		pool.join()
	
	for output_idx, output in enumerate(results):
		mean_coef[output_idx], mean_asl[output_idx], var_coef[output_idx], var_asl[output_idx] = output

	# Save the hypothesis test result
	adata.uns['scmemo']['1d_ht'] = {}
	attrs = ['design_df', 'design_matrix', 'design_matrix_cols', 'cov_column', 'mean_coef', 'mean_asl', 'var_coef', 'var_asl']
	for attr in attrs:
		adata.uns['scmemo']['1d_ht'][attr] = eval(attr)

	if not inplace:
		return adata

	
def get_1d_ht_result(adata):
	"""
		Getter function for 1d result. 
	"""
	
	result_df = pd.DataFrame()
	result_df['gene'] = adata.var.index.tolist()
	result_df['de_coef'] = adata.uns['scmemo']['1d_ht']['mean_coef']
	result_df['de_pval'] = adata.uns['scmemo']['1d_ht']['mean_asl']
	result_df['dv_coef'] = adata.uns['scmemo']['1d_ht']['var_coef']
	result_df['dv_pval'] = adata.uns['scmemo']['1d_ht']['var_asl']	
	
	return result_df
	
	