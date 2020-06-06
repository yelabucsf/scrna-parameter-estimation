import scanpy as sc
import scipy as sp
import numpy as np
import pickle as pkl

import sys
sys.path.append('/data/home/Github/scrna-parameter-estimation/schypo')
import estimator, simulate, schypo, bootstrap, util, hypothesis_test

fig_path = '/data/home/Github/scrna-parameter-estimation/figures/fig5/'
data_path = '/data/parameter_estimation/'


if __name__ == '__main__':
	
	adata = sc.read(data_path + 'interferon_filtered.h5ad')
	
	adata_ct =  adata[adata.obs.cell == 'CD14+ Monocytes'].copy()
	
	schypo.create_groups(adata_ct, label_columns=['stim', 'ind'], inplace=True, q=0.07)
	
	schypo.compute_1d_moments(
		adata_ct, inplace=True, filter_genes=True, 
		residual_var=True,filter_mean_thresh=0.00, 
		min_perc_group=0.99)
	
	target_genes = adata_ct.var.index.tolist()
	target_genes = np.random.choice(target_genes, 50, replace=False)
	
	schypo.compute_2d_moments(
		adata_ct, 
		target_genes, 
		target_genes)
	
	
	schypo.ht_2d_moments(
		adata_ct, 
		formula_like='1 + stim', 
		cov_column='stim', 
		num_cpus=6, 
		num_boot=2500)
	
	adata_ct.write(data_path + 'result_2d/mono_ifn_allgenes.h5ad')