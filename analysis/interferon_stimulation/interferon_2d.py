import scanpy as sc
import scipy as sp
import numpy as np
import pickle as pkl
import pandas as pd
import itertools

import sys
sys.path.append('/data/home/Github/scrna-parameter-estimation/dist/schypo-0.0.0-py3.7.egg')
import schypo

data_path = '/data/parameter_estimation/'


if __name__ == '__main__':
	
	adata = sc.read(data_path + 'interferon_filtered.h5ad')
	
	tf_df = pd.read_csv('DatabaseExtract_v_1.01.csv', index_col=0)
	tf_df = tf_df[tf_df['TF assessment'] == 'Known motif']
	tfs = tf_df['HGNC symbol'].tolist()
	tfs = ['NFKB1', 'JUN']
	
	adata_ct =  adata[adata.obs.cell == 'CD14+ Monocytes'].copy()
	# adata_ct.obs['stim'] = np.random.choice(cat ifnadata_ct.obs['stim'], adata_ct.shape[0])

	schypo.create_groups(adata_ct, label_columns=['ind','stim'], inplace=True, q=0.07)

	schypo.compute_1d_moments(
		adata_ct, inplace=True, filter_genes=True, 
		residual_var=True,
		filter_mean_thresh=0.07, 
		min_perc_group=0.8)
	
	genes = adata_ct.var.index.tolist()
	gene_1 = list(set(tfs) & set(genes))
	gene_2 = genes
	gene_pairs = list(itertools.product(gene_1, gene_2))
	
	schypo.compute_2d_moments(adata_ct,gene_pairs)

	schypo.ht_2d_moments(
		adata_ct, 
		formula_like='1 + stim', 
		cov_column='stim', 
		num_cpus=6, 
		num_boot=10000)
	
	adata_ct.write(data_path + 'result_2d/mono_ifn/tf_NFKB1_JUN.h5ad')
		