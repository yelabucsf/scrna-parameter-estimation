import scanpy as sc
import scipy as sp
import numpy as np
import pickle as pkl
import pandas as pd

import sys
sys.path.append('/data/home/Github/scrna-parameter-estimation/dist/schypo-0.0.0-py3.7.egg')
import schypo

data_path = '/data/parameter_estimation/'


if __name__ == '__main__':
	
	adata = sc.read(data_path + 'interferon_filtered.h5ad')
	
	tf_df = pd.read_csv('DatabaseExtract_v_1.01.csv', index_col=0)
	tf_df = tf_df[tf_df['TF assessment'] == 'Known motif']
	tfs = tf_df['HGNC symbol'].tolist()
	
	adata_ct =  adata[adata.obs.cell == 'CD14+ Monocytes'].copy()
	
	schypo.create_groups(adata_ct, label_columns=['stim', 'ind'], inplace=True, q=0.07)
	
	schypo.compute_1d_moments(
		adata_ct, inplace=True, filter_genes=True, 
		residual_var=True,filter_mean_thresh=0.025, 
		min_perc_group=0.85)
	
	target_genes = adata_ct.var.index.tolist()
	filtered_tfs = list(set(target_genes) & set(tfs))
	
	genes_per_batch = 20
	
	for batch in range(int(len(filtered_tfs)/genes_per_batch)+1):
	
		schypo.compute_2d_moments(
			adata_ct, 
			filtered_tfs[(genes_per_batch*batch):(genes_per_batch*(batch+1))], 
			target_genes)
	
		schypo.ht_2d_moments(
			adata_ct, 
			formula_like='1 + stim', 
			cov_column='stim', 
			num_cpus=90, 
			num_boot=5000)
	
		adata_ct.write(data_path + 'result_2d/mono_ifn/tf_{}.h5ad'.format(batch))
		print('Completed batch', batch)
		
		del adata_ct.uns['schypo']['2d_moments']
		del adata_ct.uns['schypo']['2d_ht']
		