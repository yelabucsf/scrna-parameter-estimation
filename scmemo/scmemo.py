"""
	scmemo.py
	
	This file contains the public facing API for using scmemo.
"""


import numpy as np
import pandas as pd

import bootstrap
import estimator
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
		adata.obs['scmemo_group'] += adata.obs[col_name]
		if idx != len(label_columns)-1:
			adata.obs['scmemo_group'] += label_delimiter
	
	# Create a dict in the uns object
	adata.uns['scmemo'] = {}
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
	residual_var=True):
	
	assert 'scmemo' in adata.uns
	
	if not inplace:
		adata = adata.copy()
		
	# Compute size factors for all groups
	adata.uns['scmemo']['size_factor'] = \
		{group:estimator._estimate_size_factor(adata.uns['scmemo']['group_cells'][group]) for group in adata.uns['scmemo']['groups']}
	
	# Compute 1d moments for all groups
	adata.uns['scmemo']['1d_moments'] = {group:estimator._poisson_1d(
		data=adata.uns['scmemo']['group_cells'][group],
		n_obs=adata.uns['scmemo']['group_cells'][group].shape[0],
		size_factor=adata.uns['scmemo']['size_factor'][group]) for group in adata.uns['scmemo']['groups']}
	
	# Create gene masks for each group
	adata.uns['scmemo']['gene_filter'] = \
		{group:(adata.uns['scmemo']['1d_moments'][group][0] > filter_mean_thresh) for group in adata.uns['scmemo']['groups']}
	
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
		
		adata.uns['scmemo']['mv_regressor'] = {
			group:estimator._fit_mv_regressor(
				mean=adata.uns['scmemo']['1d_moments'][group][0],
				var=adata.uns['scmemo']['1d_moments'][group][1]) for group in adata.uns['scmemo']['groups']}
		
		for group in adata.uns['scmemo']['groups']:
			
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
			idx2=adata.uns['scmemo']['2d_moments']['gene_idx_2'])
		
		var_1 = adata.uns['scmemo']['1d_moments'][group][1][adata.uns['scmemo']['2d_moments']['gene_idx_1']]
		var_2 = adata.uns['scmemo']['1d_moments'][group][1][adata.uns['scmemo']['2d_moments']['gene_idx_2']]
		
		corr = estimator._corr_from_cov(cov, var_1, var_2)
		
		adata.uns['scmemo']['2d_moments'][group] = (cov, corr, var_1, var_2)

	if not inplace:
		return adata

	
def bootstrap_1d_moments(adata, inplace=True, num_boot=10000):
	"""
		Computes the CI for mean and variance of each gene 
	"""
	
	if not inplace:
		adata = adata.copy()
		
	adata.uns['scmemo']['1d_ci'] = {}
	for group in adata.uns['scmemo']['groups']:
				
		# Compute 1D CIs
		mean_se, var_se, res_var_se = bootstrap._bootstrap_1d(
			data=adata.uns['scmemo']['group_cells'][group], 
			size_factor=adata.uns['scmemo']['size_factor'][group], 
			num_boot=num_boot, 
			mv_regressor=adata.uns['scmemo']['mv_regressor'][group])
		
		adata.uns['scmemo']['1d_ci'][group] = [mean_se, var_se, res_var_se]
		
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
		
		print(group)
				
		# Compute 2D CIs
		cov_se, corr_se = bootstrap._bootstrap_2d(
			data=adata.uns['scmemo']['group_cells'][group], 
			size_factor=adata.uns['scmemo']['size_factor'][group],
			gene_idxs_1=adata.uns['scmemo']['2d_moments']['gene_idx_1'],
			gene_idxs_2=adata.uns['scmemo']['2d_moments']['gene_idx_2'],
			num_boot=num_boot)
		
		adata.uns['scmemo']['2d_ci'][group] = [cov_se, corr_se]
		
	if not inplace:
		return adata
	