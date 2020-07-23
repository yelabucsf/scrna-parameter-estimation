"""
	schypo.py
	
	This file contains the public facing API for using schypo.
"""


import numpy as np
import pandas as pd
from patsy import dmatrix
import scipy.stats as stats
import sys
from joblib import Parallel, delayed
from functools import partial
import itertools

from schypo import bootstrap
from schypo import estimator
from schypo import hypothesis_test
from schypo import util
from schypo import simulate


def create_groups(
	adata,
	q,
	label_columns, 
	label_delimiter='^', 
	inplace=True,
	estimator_type='hyper_relative'
	):
	"""
		Creates discrete groups of the data, based on the columns in :label_columns:
	"""
	
	if not inplace:
		adata = adata.copy()
		
	# Create group labels
	adata.obs['schypo_group'] = 'sg' + label_delimiter
	for idx, col_name in enumerate(label_columns):
		adata.obs['schypo_group'] += adata.obs[col_name].astype(str)
		if idx != len(label_columns)-1:
			adata.obs['schypo_group'] += label_delimiter
	
	# Create a dict in the uns object
	adata.uns['schypo'] = {}
	adata.uns['schypo']['label_columns'] = label_columns
	adata.uns['schypo']['label_delimiter'] = label_delimiter
	adata.uns['schypo']['groups'] = adata.obs['schypo_group'].drop_duplicates().tolist()
	adata.uns['schypo']['q'] = q
	adata.uns['schypo']['estimator_type'] = estimator_type
	
	# Create slices of the data based on the group
	adata.uns['schypo']['group_cells'] = {group:util._select_cells(adata, group) for group in adata.uns['schypo']['groups']}
	
	if not inplace:
		return adata

def compute_1d_moments(
	adata, 
	inplace=True, 
	filter_mean_thresh=0.07, 
	min_perc_group=0.7, 
	filter_genes=True,
	num_bins=50,
	trim_percs=0.05,
	residual_var=True):
	
	assert 'schypo' in adata.uns
	
	if not inplace:
		adata = adata.copy()
		
	# Compute size factors for all groups
	naive_size_factor = estimator._estimate_size_factor(
		adata.X, 
		adata.uns['schypo']['estimator_type'], 
		total=True)
	
	# Compute residual variance over all cells
	all_1d = estimator._get_estimator_1d(adata.uns['schypo']['estimator_type'])(
		data=adata.X,
		n_obs=adata.shape[0],
		q=adata.uns['schypo']['q'],
		size_factor=naive_size_factor)
	all_res_var = estimator._residual_variance(*all_1d, estimator._fit_mv_regressor(*all_1d))
	
	# Select genes for normalization
	rv_ulim = np.quantile(all_res_var[np.isfinite(all_res_var)], trim_percs)
	all_res_var[~np.isfinite(all_res_var)] = np.inf
	rv_mask = all_res_var < rv_ulim
	
	mask = rv_mask
	print('Normalizing with {} genes'.format(mask.sum()))
	adata.uns['schypo']['least_variable_genes'] = adata.var.index[mask]
	
	size_factor = estimator._estimate_size_factor(adata.X, adata.uns['schypo']['estimator_type'], mask=mask)
	
	# Bin the size factors
	binned_stat = stats.binned_statistic(size_factor, size_factor, bins=num_bins, statistic='mean')
	bin_idx = np.clip(binned_stat[2], a_min=1, a_max=binned_stat[0].shape[0])
	approx_sf = binned_stat[0][bin_idx-1]
	max_sf = size_factor.max()
	approx_sf[size_factor == max_sf] = max_sf
	
	adata.uns['schypo']['all_size_factor'] = size_factor
	adata.uns['schypo']['all_total_size_factor'] = naive_size_factor = estimator._estimate_size_factor(adata.X, 'relative', total=True)
	adata.uns['schypo']['all_approx_size_factor'] = approx_sf
	adata.uns['schypo']['approx_size_factor'] = \
		{group:approx_sf[(adata.obs['schypo_group'] == group).values] for group in adata.uns['schypo']['groups']}
	adata.uns['schypo']['size_factor'] = \
		{group:size_factor[(adata.obs['schypo_group'] == group).values] for group in adata.uns['schypo']['groups']}
	
	# Compute 1d moments for all groups
	adata.uns['schypo']['1d_moments'] = {group:estimator._get_estimator_1d(adata.uns['schypo']['estimator_type'])(
		data=adata.uns['schypo']['group_cells'][group],
		n_obs=adata.uns['schypo']['group_cells'][group].shape[0],
		q=adata.uns['schypo']['q'],
		size_factor=adata.uns['schypo']['size_factor'][group]) for group in adata.uns['schypo']['groups']}

	# Compute 1d moments for across all cells
	adata.uns['schypo']['1d_moments']['all'] = estimator._get_estimator_1d(adata.uns['schypo']['estimator_type'])(
		data=adata.X,
		n_obs=adata.shape[0],
		q=adata.uns['schypo']['q'],
		size_factor=adata.uns['schypo']['all_size_factor'])
	
	# Create gene masks for each group
	adata.uns['schypo']['gene_filter'] = {}
	for group in adata.uns['schypo']['groups']:
		
		obs_mean = adata.uns['schypo']['group_cells'][group].mean(axis=0).A1 
		expr_filter = (obs_mean > filter_mean_thresh)
		expr_filter &= (adata.uns['schypo']['1d_moments'][group][1] > 0)
		adata.uns['schypo']['gene_filter'][group] = expr_filter
		
	# Create overall gene mask
	gene_masks = np.vstack([adata.uns['schypo']['gene_filter'][group] for group in adata.uns['schypo']['groups']])
	gene_filter_rate = gene_masks.mean(axis=0)
	overall_gene_mask = (gene_filter_rate > min_perc_group)
	adata.uns['schypo']['overall_gene_filter'] = overall_gene_mask
	adata.uns['schypo']['gene_list'] = adata.var.index[overall_gene_mask].tolist()
	
	# Filter the genes from the data matrices as well as the 1D moments
	if filter_genes:
		adata.uns['schypo']['group_cells'] = \
			{group:adata.uns['schypo']['group_cells'][group][:, overall_gene_mask] for group in adata.uns['schypo']['groups']}
		
		adata.uns['schypo']['1d_moments'] = \
			{group:[
				adata.uns['schypo']['1d_moments'][group][0][overall_gene_mask],
				adata.uns['schypo']['1d_moments'][group][1][overall_gene_mask]
				] for group in (adata.uns['schypo']['groups'] + ['all'])}
		adata._inplace_subset_var(overall_gene_mask)
	
	# Compute residual variance	
	adata.uns['schypo']['mv_regressor'] = {}
	
	adata.uns['schypo']['mv_regressor']['all'] = estimator._fit_mv_regressor(
		mean=adata.uns['schypo']['1d_moments']['all'][0],
		var=adata.uns['schypo']['1d_moments']['all'][1])
	
	for group in adata.uns['schypo']['groups']:
		
		adata.uns['schypo']['mv_regressor'][group] = estimator._fit_mv_regressor(
			mean=adata.uns['schypo']['1d_moments'][group][0],
			var=adata.uns['schypo']['1d_moments'][group][1])

		res_var = estimator._residual_variance(
			adata.uns['schypo']['1d_moments'][group][0],
			adata.uns['schypo']['1d_moments'][group][1],
			adata.uns['schypo']['mv_regressor'][group])
		adata.uns['schypo']['1d_moments'][group].append(res_var)
	
	if not inplace:
		return adata


def compute_2d_moments(adata, gene_pairs, inplace=True):
	"""
		Compute the covariance and correlation for given genes.
		This function computes the covariance and the correlation between genes in :gene_1: and genes in :gene_2:. 
	"""
	
	if not inplace:
		adata = adata.copy()
		
	# Set up the result dictionary
	adata.uns['schypo']['2d_moments'] = {}
	adata.uns['schypo']['2d_moments']['gene_pairs'] = gene_pairs
	
	# Get gene idxs 
	adata.uns['schypo']['2d_moments']['gene_idx_1'] = util._get_gene_idx(adata, [gene_1 for gene_1, gene_2 in gene_pairs])
	adata.uns['schypo']['2d_moments']['gene_idx_2'] = util._get_gene_idx(adata, [gene_2 for gene_1, gene_2 in gene_pairs])
	
	for group in adata.uns['schypo']['groups']:
		
		cov = estimator._get_estimator_cov(adata.uns['schypo']['estimator_type'])(
			data=adata.uns['schypo']['group_cells'][group], 
			n_obs=adata.uns['schypo']['group_cells'][group].shape[0], 
			q=adata.uns['schypo']['q'],
			size_factor=adata.uns['schypo']['size_factor'][group], 
			idx1=adata.uns['schypo']['2d_moments']['gene_idx_1'], 
			idx2=adata.uns['schypo']['2d_moments']['gene_idx_2'])
		
		var_1 = adata.uns['schypo']['1d_moments'][group][1][adata.uns['schypo']['2d_moments']['gene_idx_1']]
		var_2 = adata.uns['schypo']['1d_moments'][group][1][adata.uns['schypo']['2d_moments']['gene_idx_2']]
		
		corr = estimator._corr_from_cov(cov, var_1, var_2)
		
		adata.uns['schypo']['2d_moments'][group] = {'cov':cov, 'corr':corr, 'var_1':var_1, 'var_2':var_2}

	if not inplace:
		return adata


def ht_1d_moments(
	adata, 
	formula_like,
	cov_column,
	inplace=True, 
	num_boot=10000, 
	verbose=1,
	num_cpus=1):
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
	for group in adata.uns['schypo']['groups']:
		
		design_df_list.append(group.split(adata.uns['schypo']['label_delimiter'])[1:])
		Nc_list.append(adata.uns['schypo']['group_cells'][group].shape[0])
		
	# Create the design matrix from the patsy formula
	design_df = pd.DataFrame(design_df_list, columns=adata.uns['schypo']['label_columns'])
	dmat = dmatrix(formula_like, design_df)
	design_matrix_cols = dmat.design_info.column_names.copy()
	design_matrix = np.array(dmat)
	del dmat
	Nc_list = np.array(Nc_list)
	
	# Find the covariate that actually matters
	for idx, col_name in enumerate(design_matrix_cols):
		if cov_column in col_name:
			cov_idx = idx
			break
	
	# Initialize empty arrays to hold fitted coefficients and achieved significance level
	mean_coef, mean_asl, var_coef, var_asl = [np.zeros(G)*np.nan for i in range(4)]
	
	ht_funcs = []
	for idx in range(G):
		
		ht_funcs.append(
			partial(
				hypothesis_test._ht_1d,
				true_mean=[adata.uns['schypo']['1d_moments'][group][0][idx] for group in adata.uns['schypo']['groups']],
				true_res_var=[adata.uns['schypo']['1d_moments'][group][2][idx] for group in adata.uns['schypo']['groups']],
				cells=[adata.uns['schypo']['group_cells'][group][:, idx] for group in adata.uns['schypo']['groups']],
				approx_sf=[adata.uns['schypo']['approx_size_factor'][group] for group in adata.uns['schypo']['groups']],
				design_matrix=design_matrix,
				Nc_list=Nc_list,
				num_boot=num_boot,
				cov_idx=cov_idx,
				mv_fit=[adata.uns['schypo']['mv_regressor'][group] for group in adata.uns['schypo']['groups']],
				q=adata.uns['schypo']['q'],
				_estimator_1d=estimator._get_estimator_1d(adata.uns['schypo']['estimator_type'])))

	results = Parallel(n_jobs=num_cpus, verbose=verbose)(delayed(func)() for func in ht_funcs)
		
	for output_idx, output in enumerate(results):
		mean_coef[output_idx], mean_asl[output_idx], var_coef[output_idx], var_asl[output_idx] = output

	# Save the hypothesis test result
	adata.uns['schypo']['1d_ht'] = {}
	attrs = ['design_df', 'design_matrix', 'design_matrix_cols', 'cov_column', 'mean_coef', 'mean_asl', 'var_coef', 'var_asl']
	for attr in attrs:
		adata.uns['schypo']['1d_ht'][attr] = eval(attr)

	if not inplace:
		return adata

	
def ht_2d_moments(
	adata, 
	formula_like,
	cov_column,
	inplace=True, 
	num_boot=10000, 
	verbose=3,
	num_cpus=1):
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
	for group in adata.uns['schypo']['groups']:
		
		design_df_list.append(group.split(adata.uns['schypo']['label_delimiter'])[1:])
		Nc_list.append(adata.uns['schypo']['group_cells'][group].shape[0])
		
	# Create the design matrix from the patsy formula
	design_df = pd.DataFrame(design_df_list, columns=adata.uns['schypo']['label_columns'])
	dmat = dmatrix(formula_like, design_df)
	design_matrix_cols = dmat.design_info.column_names.copy()
	design_matrix = np.array(dmat)
	del dmat
	Nc_list = np.array(Nc_list)
	
	# Find the covariate that actually matters
	for idx, col_name in enumerate(design_matrix_cols):
		if cov_column in col_name:
			cov_idx = idx
			break
	
	# Get gene idxs
	gene_idx_1 = adata.uns['schypo']['2d_moments']['gene_idx_1']
	gene_idx_2 = adata.uns['schypo']['2d_moments']['gene_idx_2']
		
	# Initialize empty arrays to hold fitted coefficients and achieved significance level
	corr_coef, corr_asl = [np.zeros(gene_idx_1.shape[0])*np.nan for i in range(2)]
	
	# Create partial functions
	ht_funcs = []
	idx_list = []
	idx_mapping = {}
	
	for conv_idx in range(gene_idx_1.shape[0]):
		
		idx_1 = gene_idx_1[conv_idx]
		idx_2 = gene_idx_2[conv_idx]
		idx_set = frozenset({idx_1, idx_2})
	
		if idx_1 == idx_2: # Skip if its the same gene
			continue
			
		if idx_set in idx_mapping: # Skip if this pair of gene was already calculated
			idx_mapping[idx_set].append(conv_idx)
			continue
			
		# Save the indices
		idx_list.append((idx_1, idx_2))
		idx_mapping[idx_set] = [conv_idx]
		
		# Create the partial function
		ht_funcs.append(
			partial(
				hypothesis_test._ht_2d,
				true_corr=[adata.uns['schypo']['2d_moments'][group]['corr'][conv_idx] for group in adata.uns['schypo']['groups']],
				cells=[adata.uns['schypo']['group_cells'][group][:, [idx_1, idx_2]] for group in adata.uns['schypo']['groups']],
				approx_sf=[adata.uns['schypo']['approx_size_factor'][group] for group in adata.uns['schypo']['groups']],
				design_matrix=design_matrix,
				Nc_list=Nc_list,
				num_boot=num_boot,
				cov_idx=cov_idx,
				q=adata.uns['schypo']['q'],
				_estimator_1d=estimator._get_estimator_1d(adata.uns['schypo']['estimator_type']),
				_estimator_cov=estimator._get_estimator_cov(adata.uns['schypo']['estimator_type'])))
	
	# Parallel processing
	results = Parallel(n_jobs=num_cpus, verbose=verbose)(delayed(func)() for func in ht_funcs)
	
	for output_idx in range(len(results)):
		
		idx_1, idx_2 = idx_list[output_idx]
		
		# Fill in the value for every element that should have the same value
		for conv_idx in idx_mapping[frozenset({idx_1, idx_2})]:
			corr_coef[conv_idx], corr_asl[conv_idx] = results[output_idx]
	
	# Save the hypothesis test result
	adata.uns['schypo']['2d_ht'] = {}
	attrs = ['design_df', 'design_matrix', 'design_matrix_cols', 'cov_column', 'corr_coef', 'corr_asl']
	for attr in attrs:
		adata.uns['schypo']['2d_ht'][attr] = eval(attr)
		
	if not inplace:
		return adata
	

def get_1d_ht_result(adata):
	"""
		Getter function for 1d result. 
	"""
	
	result_df = pd.DataFrame()
	result_df['gene'] = adata.var.index.tolist()
	result_df['de_coef'] = adata.uns['schypo']['1d_ht']['mean_coef']
	result_df['de_pval'] = adata.uns['schypo']['1d_ht']['mean_asl']
	result_df['dv_coef'] = adata.uns['schypo']['1d_ht']['var_coef']
	result_df['dv_pval'] = adata.uns['schypo']['1d_ht']['var_asl']
	
	result_df['de_fdr'] = util._fdrcorrect(result_df['de_pval'].values)
	result_df['dv_fdr'] = util._fdrcorrect(result_df['dv_pval'].values)
	
	return result_df

def get_2d_ht_result(adata):
	
	result_df = pd.DataFrame(
		adata.uns['schypo']['2d_moments']['gene_pairs'],
		columns=['gene_1', 'gene_2'])
	result_df['corr_coef'] = adata.uns['schypo']['2d_ht']['corr_coef']
	result_df['corr_pval'] = adata.uns['schypo']['2d_ht']['corr_asl']

	result_df['corr_fdr'] = util._fdrcorrect(result_df['corr_pval'].values)
	
	return result_df

	