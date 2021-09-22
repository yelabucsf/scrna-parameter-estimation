"""
	main.py
	
	This file contains the public facing API for using memento.
"""


import numpy as np
import pandas as pd
from patsy import dmatrix
import scipy.stats as stats
from scipy.sparse.csr import csr_matrix
import sys
from joblib import Parallel, delayed
from functools import partial
import itertools
import logging

import memento.bootstrap as bootstrap
import memento.estimator as estimator
import memento.hypothesis_test as hypothesis_test
import memento.util as util
import memento.simulate as simulate


def setup_memento(
	adata,
	q_column,
	inplace=True,
	filter_mean_thresh=0.07, 
	trim_percent=0.1,
	shrinkage=0.5,
	num_bins=30,
	estimator_type='hyper_relative'):
	"""
		Compute size factors and the overall mean-variance regressor.
	"""
	
	print('Version 0.0.6-6')
	
	if not inplace:
		adata = adata.copy()
		
	assert adata.obs[q_column].max() < 1
	
	assert type(adata.X) == csr_matrix, 'please make sure that adata.X is a scipy CSR matrix'
	
	# Setup the memento dictionary in uns
	adata.uns['memento'] = {}
	adata.uns['memento']['q_column'] = q_column
	adata.uns['memento']['all_q'] = adata.obs[q_column].values.mean()
	adata.uns['memento']['estimator_type'] = estimator_type
	adata.uns['memento']['filter_mean_thresh'] = filter_mean_thresh
	adata.uns['memento']['num_bins'] = num_bins
		
	# Compute size factors for all groups
	naive_size_factor = estimator._estimate_size_factor(
		adata.X, 
		adata.uns['memento']['estimator_type'], 
		total=True,
		shrinkage=0.0)
	
	# Compute residual variance over all cells with naive size factor
	all_m, all_v = estimator._get_estimator_1d(adata.uns['memento']['estimator_type'])(
		data=adata.X,
		n_obs=adata.shape[0],
		q=adata.uns['memento']['all_q'],
		size_factor=naive_size_factor)
	all_m[adata.X.mean(axis=0).A1 < filter_mean_thresh] = 0 # mean filter
	all_res_var = estimator._residual_variance(all_m, all_v, estimator._fit_mv_regressor(all_m, all_v))
	
	# Select genes for normalization
	rv_ulim = np.quantile(all_res_var[np.isfinite(all_res_var)], trim_percent)
	all_res_var[~np.isfinite(all_res_var)] = np.inf
	rv_mask = all_res_var < rv_ulim
	mask = rv_mask
	adata.uns['memento']['least_variable_genes'] = adata.var.index[mask].tolist()
	
	# Re-estimate the size factor
	size_factor = estimator._estimate_size_factor(
		adata.X, 
		adata.uns['memento']['estimator_type'], 
		mask=mask,
		shrinkage=shrinkage)
	adata.obs['memento_size_factor'] = size_factor
	
	# Re-estimate the mean-variance regressor from mean, variance across all cells
	all_m, all_v = estimator._get_estimator_1d(adata.uns['memento']['estimator_type'])(
		data=adata.X,
		n_obs=adata.shape[0],
		q=adata.uns['memento']['all_q'],
		size_factor=size_factor)
	adata.uns['memento']['all_1d_moments'] = [all_m, all_v]


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
	
	# Clean up the uns object
	# These are the keys that should be present after computing size factors.
	# All others are from prior manipulations (bug?)
# 	key_list = ['q_column', 'all_q', 'estimator_type', 'filter_mean_thresh', 'num_bins', 'least_variable_genes', 'all_1d_moments']
# 	for key in list(adata.uns['memento'].keys()):
# 		if key not in key_list:
# 			del adata.uns['memento'][key]
			
	# Create group labels
	adata.obs['memento_group'] = 'sg' + label_delimiter
	for idx, col_name in enumerate(label_columns):
		adata.obs['memento_group'] += adata.obs[col_name].astype(str)
		if idx != len(label_columns)-1:
			adata.obs['memento_group'] += label_delimiter
	
	# Create a dict in the uns object
	adata.uns['memento']['label_columns'] = label_columns
	adata.uns['memento']['label_delimiter'] = label_delimiter
	adata.uns['memento']['groups'] = adata.obs['memento_group'].drop_duplicates().tolist()
	adata.uns['memento']['q'] = adata.obs[adata.uns['memento']['q_column']].values
	
	# Create slices of the data based on the group
	adata.uns['memento']['group_cells'] = {group:util._select_cells(adata, group) for group in adata.uns['memento']['groups']}
	
	# For each slice, get mean q
	adata.uns['memento']['group_q'] = {group:adata.uns['memento']['q'][(adata.obs['memento_group'] == group).values].mean() \
		for group in adata.uns['memento']['groups']}
	
	if not inplace:
		return adata
	

def _bin_size_factor(adata):
	"""
		Organize the size factors into different groups
	"""
	size_factor = adata.obs['memento_size_factor'].values
	binned_stat = stats.binned_statistic(size_factor, size_factor, bins=adata.uns['memento']['num_bins'], statistic='mean')
	bin_idx = np.clip(binned_stat[2], a_min=1, a_max=binned_stat[0].shape[0])
	approx_sf = binned_stat[0][bin_idx-1]
	max_sf = size_factor.max()
	approx_sf[size_factor == max_sf] = max_sf

	adata.uns['memento']['all_total_size_factor'] = estimator._estimate_size_factor(adata.X, 'relative', total=True)
	adata.uns['memento']['all_approx_size_factor'] = approx_sf
	adata.uns['memento']['approx_size_factor'] = \
		{group:approx_sf[(adata.obs['memento_group'] == group).values] for group in adata.uns['memento']['groups']}
	adata.uns['memento']['size_factor'] = \
		{group:size_factor[(adata.obs['memento_group'] == group).values] for group in adata.uns['memento']['groups']}
	
	
def compute_1d_moments(
	adata, 
	inplace=True, 
	min_perc_group=0.7, 
	filter_genes=True,
	gene_list=None):
	"""
		Compute the mean, variance, and residual variance in each group
	"""
	
	assert 'memento' in adata.uns
	
	if not inplace:
		adata = adata.copy()
	
	if 'size_factor' not in adata.uns['memento'].keys():
		_bin_size_factor(adata)
	
	# Compute 1d moments for all groups
	adata.uns['memento']['1d_moments'] = {group:estimator._get_estimator_1d(adata.uns['memento']['estimator_type'])(
		data=adata.uns['memento']['group_cells'][group],
		n_obs=adata.uns['memento']['group_cells'][group].shape[0],
		q=adata.uns['memento']['group_q'][group],
		size_factor=adata.uns['memento']['size_factor'][group]) for group in adata.uns['memento']['groups']}
	
	# Create gene masks for each group
	adata.uns['memento']['gene_filter'] = {}
	for group in adata.uns['memento']['groups']:
		
		obs_mean = adata.uns['memento']['group_cells'][group].mean(axis=0).A1 
		expr_filter = (obs_mean > adata.uns['memento']['filter_mean_thresh'])
		expr_filter &= (adata.uns['memento']['1d_moments'][group][1] > 0)
		adata.uns['memento']['gene_filter'][group] = expr_filter
		
	# Create overall gene mask
	gene_masks = np.vstack([adata.uns['memento']['gene_filter'][group] for group in adata.uns['memento']['groups']])
	gene_filter_rate = gene_masks.mean(axis=0)
	overall_gene_mask = (gene_filter_rate > min_perc_group)

	# If a gene list is given, use that list instead
	if gene_list is not None:
		assert type(gene_list) == list
		overall_gene_mask = np.in1d(adata.var.index.values, gene_list)

	# Do the filtering
	adata.uns['memento']['overall_gene_filter'] = overall_gene_mask
	adata.uns['memento']['gene_list'] = adata.var.index[overall_gene_mask].tolist()
		
	# Filter the genes from the data matrices as well as the 1D moments
	if filter_genes:
		adata.uns['memento']['group_cells'] = \
			{group:adata.uns['memento']['group_cells'][group][:, overall_gene_mask] for group in adata.uns['memento']['groups']}
		
		adata.uns['memento']['1d_moments'] = \
			{group:[
				adata.uns['memento']['1d_moments'][group][0][overall_gene_mask],
				adata.uns['memento']['1d_moments'][group][1][overall_gene_mask]
				] for group in (adata.uns['memento']['groups'])}
		adata._inplace_subset_var(overall_gene_mask)
	
	# Estimate the residual variance transformer for all cells
	mean_list = []
	var_list = []
	for group in adata.uns['memento']['groups']:
		mean_list.append(adata.uns['memento']['1d_moments'][group][0])
		var_list.append(adata.uns['memento']['1d_moments'][group][1])
	mean_concat = np.concatenate(mean_list)
	var_concat = np.concatenate(var_list)
	adata.uns['memento']['mv_regressor'] = {'all':estimator._fit_mv_regressor(mean_concat, var_concat)}
	
	# Estimate the residual variance transformer for each group
	for group in adata.uns['memento']['groups']:
		m = adata.uns['memento']['1d_moments'][group][0]
		v = adata.uns['memento']['1d_moments'][group][1]
		adata.uns['memento']['mv_regressor'][group] = estimator._fit_mv_regressor(mean_concat, var_concat)
	
	# Compute the residual variance
	for group in adata.uns['memento']['groups']:
		
		res_var = estimator._residual_variance(
			adata.uns['memento']['1d_moments'][group][0],
			adata.uns['memento']['1d_moments'][group][1],
			adata.uns['memento']['mv_regressor'][group])
		
		adata.uns['memento']['1d_moments'][group].append(res_var)
			
	if not inplace:
		return adata
	
def get_corr_matrix(adata):
	"""
		Computes the all by all correlation matrix.
	"""
	
	corr_matrices = {}
	
	for group in adata.uns['memento']['groups']:
		corr_matrices[group] = estimator._hyper_corr_symmetric(
			data=adata.uns['memento']['group_cells'][group], 
			n_obs=adata.uns['memento']['group_cells'][group].shape[0], 
			size_factor=adata.uns['memento']['size_factor'][group], 
			q=adata.uns['memento']['group_q'][group], 
			var=adata.uns['memento']['1d_moments'][group][1], 
			idx1=None, 
			idx2=None)
		
	return corr_matrices
	
def compute_2d_moments(adata, gene_pairs, inplace=True):
	"""
		Compute the covariance and correlation for given genes.
		This function computes the covariance and the correlation between genes in :gene_1: and genes in :gene_2:. 
	"""
	
	if not inplace:
		adata = adata.copy()
		
	if 'size_factor' not in adata.uns['memento'].keys():
		_bin_size_factor(adata)
		
	# Set up the result dictionary
	adata.uns['memento']['2d_moments'] = {}
	adata.uns['memento']['2d_moments']['gene_pairs'] = gene_pairs
	
	# Get gene idxs
	n_pairs = len(gene_pairs)
	mapping = dict(zip(adata.var.index.values, np.arange(adata.var.shape[0])))
	adata.uns['memento']['2d_moments']['gene_idx_1'] = np.zeros(n_pairs, dtype=int)
	adata.uns['memento']['2d_moments']['gene_idx_2'] = np.zeros(n_pairs, dtype=int)
	pair_idx = 0
	for gene_1, gene_2 in gene_pairs:
		adata.uns['memento']['2d_moments']['gene_idx_1'][pair_idx] = mapping[gene_1]
		adata.uns['memento']['2d_moments']['gene_idx_2'][pair_idx] = mapping[gene_2]
		pair_idx += 1
		
	for group in adata.uns['memento']['groups']:
		
		cov = estimator._get_estimator_cov(adata.uns['memento']['estimator_type'])(
			data=adata.uns['memento']['group_cells'][group], 
			n_obs=adata.uns['memento']['group_cells'][group].shape[0], 
			q=adata.uns['memento']['group_q'][group],
			size_factor=adata.uns['memento']['size_factor'][group], 
			idx1=adata.uns['memento']['2d_moments']['gene_idx_1'], 
			idx2=adata.uns['memento']['2d_moments']['gene_idx_2'])
		
		var_1 = adata.uns['memento']['1d_moments'][group][1][adata.uns['memento']['2d_moments']['gene_idx_1']]
		var_2 = adata.uns['memento']['1d_moments'][group][1][adata.uns['memento']['2d_moments']['gene_idx_2']]
		
		corr = estimator._corr_from_cov(cov, var_1, var_2)
		
		adata.uns['memento']['2d_moments'][group] = {'cov':cov, 'corr':corr, 'var_1':var_1, 'var_2':var_2}

	if not inplace:
		return adata


def ht_1d_moments(
	adata, 
	formula_like,
	treatment_col,
	inplace=True, 
	num_boot=10000, 
	verbose=1,
	num_cpus=1,
	**kwargs):
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
	for group in adata.uns['memento']['groups']:
		
		design_df_list.append(group.split(adata.uns['memento']['label_delimiter'])[1:])
		Nc_list.append(adata.uns['memento']['group_cells'][group].shape[0])
		
	# Create the design matrix from the patsy formula
	design_df = pd.DataFrame(design_df_list, columns=adata.uns['memento']['label_columns'])
	dmat = dmatrix(formula_like, design_df)
	design_matrix_cols = dmat.design_info.column_names.copy()
	design_matrix = np.array(dmat)
	del dmat
	Nc_list = np.array(Nc_list)
	
	# Find the covariate that actually matters
	treatment_idx = []
	for idx, col_name in enumerate(design_matrix_cols):
		if treatment_col in col_name:
			treatment_idx.append(idx)
	assert len(treatment_idx) > 0, 'could not find treatment column'
	
	# Initialize empty arrays to hold fitted coefficients and achieved significance level
	mean_coef, mean_se, mean_asl, var_coef, var_se, var_asl = [np.zeros(G)*np.nan for i in range(6)]
	
	ht_funcs = []
	for idx in range(G):
		
		ht_funcs.append(
			partial(
				hypothesis_test._ht_1d,
				true_mean=[adata.uns['memento']['1d_moments'][group][0][idx] for group in adata.uns['memento']['groups']],
				true_res_var=[adata.uns['memento']['1d_moments'][group][2][idx] for group in adata.uns['memento']['groups']],
				cells=[adata.uns['memento']['group_cells'][group][:, idx] for group in adata.uns['memento']['groups']],
				approx_sf=[adata.uns['memento']['approx_size_factor'][group] for group in adata.uns['memento']['groups']],
				design_matrix=design_matrix,
				Nc_list=Nc_list,
				num_boot=num_boot,
				treatment_idx=treatment_idx,
				mv_fit=[adata.uns['memento']['mv_regressor'][group] for group in adata.uns['memento']['groups']],
				q=[adata.uns['memento']['group_q'][group] for group in adata.uns['memento']['groups']],
				_estimator_1d=estimator._get_estimator_1d(adata.uns['memento']['estimator_type']),
				**kwargs))

	results = Parallel(n_jobs=num_cpus, verbose=verbose)(delayed(func)() for func in ht_funcs)
		
	for output_idx, output in enumerate(results):
		mean_coef[output_idx], mean_se[output_idx], mean_asl[output_idx], var_coef[output_idx], var_se[output_idx], var_asl[output_idx] = output

	# Save the hypothesis test result
	adata.uns['memento']['1d_ht'] = {}
	attrs = ['design_df', 'design_matrix', 'design_matrix_cols', 'treatment_col', 'mean_coef', 'mean_se','mean_asl', 'var_coef', 'var_se','var_asl']
	for attr in attrs:
		adata.uns['memento']['1d_ht'][attr] = eval(attr)

	if not inplace:
		return adata

	
def ht_2d_moments(
	adata, 
	formula_like,
	treatment_col,
	inplace=True, 
	num_boot=10000, 
	verbose=3,
	num_cpus=1,
	**kwargs):
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
	for group in adata.uns['memento']['groups']:
		
		design_df_list.append(group.split(adata.uns['memento']['label_delimiter'])[1:])
		Nc_list.append(adata.uns['memento']['group_cells'][group].shape[0])
		
	# Create the design matrix from the patsy formula
	design_df = pd.DataFrame(design_df_list, columns=adata.uns['memento']['label_columns'])
	dmat = dmatrix(formula_like, design_df)
	design_matrix_cols = dmat.design_info.column_names.copy()
	design_matrix = np.array(dmat)
	del dmat
	Nc_list = np.array(Nc_list)
	
	# Find the covariate that actually matters
	treatment_idx = []
	for idx, col_name in enumerate(design_matrix_cols):
		if treatment_col in col_name:
			treatment_idx.append(idx)
	assert len(treatment_idx) > 0, 'could not find treatment column'
	
	# Get gene idxs
	gene_idx_1 = adata.uns['memento']['2d_moments']['gene_idx_1']
	gene_idx_2 = adata.uns['memento']['2d_moments']['gene_idx_2']
		
	# Initialize empty arrays to hold fitted coefficients and achieved significance level
	corr_coef, corr_se, corr_asl = [np.zeros(gene_idx_1.shape[0])*np.nan for i in range(3)]
		
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
				true_corr=[adata.uns['memento']['2d_moments'][group]['corr'][conv_idx] for group in adata.uns['memento']['groups']],
				cells=[adata.uns['memento']['group_cells'][group][:, [idx_1, idx_2]] for group in adata.uns['memento']['groups']],
				approx_sf=[adata.uns['memento']['approx_size_factor'][group] for group in adata.uns['memento']['groups']],
				design_matrix=design_matrix,
				Nc_list=Nc_list,
				num_boot=num_boot,
				treatment_idx=treatment_idx,
				q=[adata.uns['memento']['group_q'][group] for group in adata.uns['memento']['groups']],
				_estimator_1d=estimator._get_estimator_1d(adata.uns['memento']['estimator_type']),
				_estimator_cov=estimator._get_estimator_cov(adata.uns['memento']['estimator_type']),
				**kwargs))
	
	# Parallel processing
	results = Parallel(n_jobs=num_cpus, verbose=verbose)(delayed(func)() for func in ht_funcs)
	
	for output_idx in range(len(results)):
		
		idx_1, idx_2 = idx_list[output_idx]
		
		# Fill in the value for every element that should have the same value
		for conv_idx in idx_mapping[frozenset({idx_1, idx_2})]:
			corr_coef[conv_idx], corr_se[conv_idx], corr_asl[conv_idx] = results[output_idx]
	
	# Save the hypothesis test result
	adata.uns['memento']['2d_ht'] = {}
	attrs = ['design_df', 'design_matrix', 'design_matrix_cols', 'treatment_col', 'corr_coef', 'corr_se','corr_asl']
	for attr in attrs:
		adata.uns['memento']['2d_ht'][attr] = eval(attr)
		
	if not inplace:
		return adata


def get_1d_moments(adata, groupby=None):
	"""
		Getter function for 1d moments.
		If groupby is used, take the mean of the moments weighted by cell counts
	"""
	
	moment_mean_df = pd.DataFrame()
	moment_mean_df['gene'] = adata.var.index.tolist()
	moment_var_df = pd.DataFrame()
	moment_var_df['gene'] = adata.var.index.tolist()
	
	cell_counts = {k:v.shape[0] for k,v in adata.uns['memento']['group_cells'].items()}
	for group, val in adata.uns['memento']['1d_moments'].items():
		if group == 'all':
			continue
		moment_mean_df[group] = np.log(val[0])
		moment_var_df[group] = np.log(val[2])
	
	if groupby is None:
		return moment_mean_df, moment_var_df, cell_counts
	
	if groupby != 'ALL':
		unique_groupby = adata.obs[groupby].astype(str).drop_duplicates().values
	else:
		unique_groupby = ['sg']
	groupby_mean = {k:0 for k in unique_groupby}
	groupby_var = {k:0 for k in unique_groupby}
	groupby_mean_count = {k:0 for k in unique_groupby}
	groupby_var_count = {k:0 for k in unique_groupby}
	
	groupby_mean_df = pd.DataFrame()
	groupby_mean_df['gene'] = adata.var.index.tolist()
	groupby_var_df = pd.DataFrame()
	groupby_var_df['gene'] = adata.var.index.tolist()

	for key in unique_groupby:
		for group, val in adata.uns['memento']['1d_moments'].items():
			if group == 'all':
				continue
				
			if key in group: # stringfied
				
				m = np.log(val[0])
				v = np.log(val[2])
				m[np.isnan(m)] = 0
				v[np.isnan(v)] = 0
				
				groupby_mean[key] += m*cell_counts[group]
				groupby_mean_count[key] += (val[0] > 0)*cell_counts[group]

				groupby_var[key] += v*cell_counts[group]
				groupby_var_count[key] += (val[2] > 0)*cell_counts[group]

		groupby_mean[key] /= groupby_mean_count[key]
		groupby_var[key] /= groupby_var_count[key]
	
		groupby_mean_df[groupby + '_' + key] = groupby_mean[key]
		groupby_var_df[groupby + '_' + key] = groupby_var[key]
	
	return groupby_mean_df.copy(), groupby_var_df.copy()


def get_2d_moments(adata, groupby=None):
	"""
		Getter function for 1d moments.
		If groupby is used, take the mean of the moments weighted by cell counts
	"""
	
	moment_corr_df = pd.DataFrame(
		adata.uns['memento']['2d_moments']['gene_pairs'],
		columns=['gene_1', 'gene_2'])
	
	cell_counts = {k:v.shape[0] for k,v in adata.uns['memento']['group_cells'].items()}
	for group, val in adata.uns['memento']['2d_moments'].items():
		if 'sg^' not in group:
			continue
		moment_corr_df[group] = val['corr']
	
	if groupby is None:
		return moment_corr_df, cell_counts
	
	if groupby != 'ALL':
		unique_groupby = adata.obs[groupby].astype(str).drop_duplicates().values
	else:
		unique_groupby = ['sg']
	groupby_corr = {k:0 for k in unique_groupby}
	groupby_corr_count = {k:0 for k in unique_groupby}
	
	groupby_corr_df = pd.DataFrame()
	groupby_corr_df['gene_1'] = moment_corr_df['gene_1']
	groupby_corr_df['gene_2'] = moment_corr_df['gene_2']
	
	for key in unique_groupby:
		for group, val in adata.uns['memento']['2d_moments'].items():
			if 'sg^' not in group:
				continue
				
			if key in group: #stringfied
				
				c = val['corr']
				valid = ~np.isnan(c)
				c[np.isnan(c)] = 0
				
				groupby_corr[key] += c*cell_counts[group]
				groupby_corr_count[key] += valid*cell_counts[group]

		groupby_corr[key] /= groupby_corr_count[key]
		groupby_corr_df[groupby + '_' + key] = groupby_corr[key]
		
	return groupby_corr_df.copy()

	
def get_1d_ht_result(adata):
	"""
		Getter function for 1d HT result. 
	"""
	
	result_df = pd.DataFrame()
	result_df['gene'] = adata.var.index.tolist()
	result_df['de_coef'] = adata.uns['memento']['1d_ht']['mean_coef']
	if 'mean_se' in adata.uns['memento']['1d_ht']:
		result_df['de_se'] = adata.uns['memento']['1d_ht']['mean_se']
	result_df['de_pval'] = adata.uns['memento']['1d_ht']['mean_asl']
	result_df['dv_coef'] = adata.uns['memento']['1d_ht']['var_coef']
	if 'var_se' in adata.uns['memento']['1d_ht']:
		result_df['dv_se'] = adata.uns['memento']['1d_ht']['var_se']
	result_df['dv_pval'] = adata.uns['memento']['1d_ht']['var_asl']
		
	return result_df


def get_2d_ht_result(adata):
	"""
		Getter function for 2d HT result
	"""
	
	result_df = pd.DataFrame(
		adata.uns['memento']['2d_moments']['gene_pairs'],
		columns=['gene_1', 'gene_2'])
	result_df['corr_coef'] = adata.uns['memento']['2d_ht']['corr_coef']
	result_df['corr_se'] = adata.uns['memento']['2d_ht']['corr_se']
	result_df['corr_pval'] = adata.uns['memento']['2d_ht']['corr_asl']
	
	return result_df


def prepare_to_save(adata, keep=False):
	"""
		pickle all objects that aren't compatible with scanpy write
	"""
	
	for group in adata.uns['memento']['groups'] + ['all']:
		
		if not keep:
			del adata.uns['memento']['mv_regressor'][group]
		else:
			adata.uns['memento']['mv_regressor'][group] = str(pkl.dumps(adata.uns['memento']['mv_regressor'][group]))