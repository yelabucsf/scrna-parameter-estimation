import scanpy as sc
import seaborn as sns
import pandas as pd
import numpy as np
import itertools

import argparse

import sys
sys.path.append('/home/ssm-user/Github/scrna-parameter-estimation/dist/memento-0.0.8-py3.8.egg')
import memento

def run_onek_sampled(pop, inds):
	
	data_path  = '/data_volume/memento/lupus/'

	pos = pd.read_csv(data_path + 'mateqtl_input/sampled_pos/{}_{}.tsv'.format(pop, inds), sep='\t', index_col=0)
	cov = pd.read_csv(data_path + 'mateqtl_input/sampled_cov/{}_{}.tsv'.format(pop, inds), sep='\t', index_col=0).T
	
	onek_replication = pd.read_csv(data_path + 'filtered_onek_eqtls.csv')
	cts = onek_replication.cg_cov.drop_duplicates().tolist()
	
	for ct in cts:

		adata = sc.read(data_path + 'single_cell/{}_{}.h5ad'.format(pop, ct))
		adata = adata[adata.obs.ind_cov.isin(pos.columns)].copy()

		adata.obs['capture_rate'] = 0.1
		memento.setup_memento(adata, q_column='capture_rate', trim_percent=0.1, filter_mean_thresh=0.01)
# 		adata.obs['memento_size_factor'] = 1.0
		memento.create_groups(adata, label_columns=['ind_cov'])

		cov_df = cov.loc[[x[3:] for x in adata.uns['memento']['groups']]]

		donor_df = pos[[x[3:] for x in adata.uns['memento']['groups']]].T

		gene_snp_pairs = onek_replication.query('cg_cov == "{}"'.format(ct))
		memento.compute_1d_moments(adata, min_perc_group=.9, gene_list=gene_snp_pairs.gene.drop_duplicates().tolist())

		gene_to_snp = dict(gene_snp_pairs[gene_snp_pairs.gene.isin(adata.var.index)].groupby('gene').rsid.apply(list))

		memento.ht_1d_moments(
			adata, 
			covariate=cov_df,
			treatment=donor_df,
			treatment_for_gene=gene_to_snp,
			num_boot=5000, 
			verbose=1,
			num_cpus=93,
			resampling='bootstrap',
			approx=True,
			resample_rep=True)

	#     adata.write(data_path + 'memento_1k/{}.h5ad'.format(ct))
		memento_result = memento.get_1d_ht_result(adata)
		memento_result.to_csv(data_path + 'memento_1k/{}_{}_{}.csv'.format(pop, ct, inds), index=False)
		pseudobulk = pd.read_csv(data_path + 'pseudobulk/{}_{}.csv'.format(pop, ct), index_col=0, sep='\t')
		pseudobulk = pseudobulk.loc[pos.columns, memento_result['gene'].drop_duplicates().tolist()]
		pseudobulk.T.to_csv(data_path + 'mateqtl_input/sampled_pseudobulk/{}_{}_{}.csv'.format(pop, ct, inds), sep='\t')

def run_full(pop, window='100kb', num_blocks=5):
	
	data_path  = '/data_volume/memento/lupus/'
	
	pos = pd.read_csv(data_path + 'mateqtl_input/{}_genos.tsv'.format(pop), sep='\t', index_col=0)
	cov = pd.read_csv(data_path + 'mateqtl_input/{}_mateqtl_cov.txt'.format(pop), sep='\t', index_col=0).T
	
	gene_snp_pairs = pd.read_csv(data_path + 'mateqtl_input/{}/gene_snp_pairs_hg19_100kb.csv'.format(pop)).query('rsid in @pos.index')
	gene_snp_pairs = gene_snp_pairs.sample(frac=1)
	
	cts = ['T4', 'cM', 'ncM', 'T8', 'B', 'NK']

	blocksize=int(gene_snp_pairs.shape[0]/num_blocks)
	print(blocksize)
	
	for ct in cts:
		
		print(ct)

		adata = sc.read(data_path + 'single_cell/{}_{}.h5ad'.format(pop, ct))
		adata = adata[adata.obs.ind_cov.isin(pos.columns)].copy()

		adata.obs['capture_rate'] = 0.1
		memento.setup_memento(adata, q_column='capture_rate', trim_percent=0.1, filter_mean_thresh=0.01)
# 		adata.obs['memento_size_factor'] = 1.0
		memento.create_groups(adata, label_columns=['ind_cov'])

		cov_df = cov.loc[[x[3:] for x in adata.uns['memento']['groups']]]

		donor_df = pos[[x[3:] for x in adata.uns['memento']['groups']]].T
		
		for block in range(num_blocks):
			subset_gene_snp_pairs = gene_snp_pairs.iloc[(blocksize*block):(blocksize*(block+1)), :]
			
			memento.compute_1d_moments(adata, min_perc_group=.2, gene_list=subset_gene_snp_pairs.gene.drop_duplicates().tolist())
			
			gene_to_snp = dict(subset_gene_snp_pairs[subset_gene_snp_pairs.gene.isin(adata.var.index)].groupby('gene').rsid.apply(list))

			memento.ht_1d_moments(
				adata, 
				covariate=cov_df,
				treatment=donor_df,
				treatment_for_gene=gene_to_snp,
				num_boot=3000, 
				verbose=1,
				num_cpus=80,
				resampling='bootstrap',
				approx=True,
				resample_rep=True)

		#     adata.write(data_path + 'memento_1k/{}.h5ad'.format(ct))
			memento_result = memento.get_1d_ht_result(adata)
			memento_result.to_csv(data_path + 'memento_full/{}/{}_{}_block_{}.csv'.format(window,pop, ct, block), index=False)