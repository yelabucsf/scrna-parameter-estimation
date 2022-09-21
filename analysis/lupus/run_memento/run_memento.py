import scanpy as sc
import seaborn as sns
import pandas as pd
import numpy as np
import itertools

import argparse

import sys
sys.path.append('/home/ssm-user/Github/scrna-parameter-estimation/dist/memento-0.0.9-py3.8.egg')
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

def run_full(pop, window='100kb', num_blocks=5, parameter='mean'):
	
	data_path  = '/data_volume/memento/lupus/'
	
	pos = pd.read_csv(data_path + 'mateqtl_input/{}_genos.tsv'.format(pop), sep='\t', index_col=0)
	cov = pd.read_csv(data_path + 'mateqtl_input/{}_mateqtl_cov.txt'.format(pop), sep='\t', index_col=0).T
	
	gene_snp_pairs = pd.read_csv(data_path + 'mateqtl_input/{}/gene_snp_pairs_hg19_100kb.csv'.format(pop)).query('rsid in @pos.index')
	gene_snp_pairs = gene_snp_pairs.sample(frac=1)
	
	cts = ['T4', 'cM','ncM', 'T8', 'B', 'NK']

	blocksize=int(gene_snp_pairs.shape[0]/num_blocks)
	print(blocksize)
	
	for ct in cts:
		
		print(ct)

		adata = sc.read(data_path + 'single_cell/{}_{}.h5ad'.format(pop, ct))
		donor_counts = adata.obs['ind_cov'].value_counts()
		if parameter=='mean':
			filter_condition = adata.obs.ind_cov.isin(pos.columns)
		else:
			filter_condition = adata.obs.ind_cov.isin(pos.columns) & adata.obs.ind_cov.isin(donor_counts[donor_counts > 100].index)
		adata = adata[filter_condition].copy()

		adata.obs['capture_rate'] = 0.1
		if parameter=='mean':
			memento.setup_memento(adata, q_column='capture_rate', trim_percent=0.1, filter_mean_thresh=0.01, estimator_type='mean_only')
		else:
			memento.setup_memento(adata, q_column='capture_rate', trim_percent=0.1, filter_mean_thresh=0.05)
# 		adata.obs['memento_size_factor'] = 1.0
		memento.create_groups(adata, label_columns=['ind_cov'])

		cov_df = cov.loc[[x[3:] for x in adata.uns['memento']['groups']]]

		donor_df = pos[[x[3:] for x in adata.uns['memento']['groups']]].T
		
		for block in range(num_blocks):
			print(block)
			subset_gene_snp_pairs = gene_snp_pairs.iloc[(blocksize*block):(blocksize*(block+1)), :]
			
			memento.compute_1d_moments(adata, min_perc_group=.2, gene_list=subset_gene_snp_pairs.gene.drop_duplicates().tolist())
			
			gene_to_snp = dict(subset_gene_snp_pairs[subset_gene_snp_pairs.gene.isin(adata.var.index)].groupby('gene').rsid.apply(list))

			memento.ht_1d_moments(
				adata, 
				covariate=cov_df,
				treatment=donor_df,
				treatment_for_gene=gene_to_snp,
				num_boot=3000 if parameter=='mean' else 5000, 
				verbose=1,
				num_cpus=80,
				resampling='bootstrap',
				approx=False,
				resample_rep=True)

		#     adata.write(data_path + 'memento_1k/{}.h5ad'.format(ct))
			memento_result = memento.get_1d_ht_result(adata)
			if parameter=='mean':
				memento_result.to_csv(data_path + 'memento_full/{}/{}_{}_block_{}.csv'.format(window,pop, ct, block), index=False)
			if parameter=='variability':
				memento_result.to_csv(data_path + 'memento_full/{}/{}_{}_block_{}_variability_no_approx.csv'.format(window,pop, ct, block), index=False)
				
				
def run_full_coexpression(pop, window='100kb', num_blocks=5):
	
	data_path  = '/data_volume/memento/lupus/'
	
	# Read genetics data
	pos = pd.read_csv(data_path + 'mateqtl_input/{}_genos.tsv'.format(pop), sep='\t', index_col=0)
	cov = pd.read_csv(data_path + 'mateqtl_input/{}_mateqtl_cov.txt'.format(pop), sep='\t', index_col=0).T
	
	# Read encode data to filter TFs
	encode_meta = pd.read_csv('../../cd4_cropseq/encode_tf/metadata.tsv', sep='\t', header=0)
	encode_files = pd.read_csv('../../cd4_cropseq/encode_tf/files.txt', sep='\t', header=None)
	encode_meta = encode_meta[encode_meta['Output type'].isin([ 'IDR thresholded peaks', 'optimal IDR thresholded peaks']) & (encode_meta['File assembly'] == 'GRCh38')]
	encode_meta['target'] = encode_meta['Experiment target'].str.split('-').str[0]
	encode_meta = encode_meta.sort_values('Output type', ascending=False)
	encode_meta = encode_meta[encode_meta['Audit ERROR'].isnull() & encode_meta['Audit NOT_COMPLIANT'].isnull()]
	encode_tfs = encode_meta.target.tolist()
	
	cts = ['T4', 'cM','ncM', 'T8', 'B', 'NK']
	
	for ct in cts:
		
		print(ct)

		adata = sc.read(data_path + 'single_cell/{}_{}.h5ad'.format(pop, ct))
		donor_counts = adata.obs['ind_cov'].value_counts()
		filter_condition = adata.obs.ind_cov.isin(pos.columns) & adata.obs.ind_cov.isin(donor_counts[donor_counts > 100].index)
		adata = adata[filter_condition].copy()
		adata.obs['capture_rate'] = 0.1
		memento.setup_memento(adata, q_column='capture_rate', trim_percent=0.1, filter_mean_thresh=0.07)
		memento.create_groups(adata, label_columns=['ind_cov'])

		cov_df = cov.loc[[x[3:] for x in adata.uns['memento']['groups']]]
		donor_df = pos[[x[3:] for x in adata.uns['memento']['groups']]].T
				
		# Get all genes of interest
		eqtls = pd.read_csv(data_path + 'full_analysis/memento/100kb/{}_{}.csv'.format(pop, ct))\
			.query('FDR < 0.01 & (beta < -0.1 | beta > 0.1)')\
			.rename(columns={'gene':'gene_2'})[['SNP', 'gene_2']].copy()
		eGenes = eqtls['gene_2'].drop_duplicates().tolist()
		all_genes_of_interest = list(set(eGenes + encode_tfs))
		
		# Get a list of filtered TFs
		memento.compute_1d_moments(adata, min_perc_group=.5, gene_list=all_genes_of_interest)
		filtered_tfs = list(set(encode_tfs) & set(adata.var.index))
		print('num TFs', len(filtered_tfs))
		
		# Create gene_1, gene_2, SNP triplets
		gene_snp_pairs = eqtls[eqtls.gene_2.isin(adata.var.index)].copy()
		gene_snp_pairs['gene_1'] = [filtered_tfs]*len(gene_snp_pairs)
		gene_snp_pairs = gene_snp_pairs.explode('gene_1').sample(frac=1, ignore_index=True)
		
		blocksize=int(gene_snp_pairs.shape[0]/num_blocks)
		print('blocksize', blocksize)
		
		print('number of tests', gene_snp_pairs.shape[0])
		
		for block in range(num_blocks):
			print('block', block)
			subset_gene_snp_pairs = gene_snp_pairs.iloc[(blocksize*block):(blocksize*(block+1)), :]
						
			gene_to_snp = dict(subset_gene_snp_pairs.groupby(['gene_1', 'gene_2']).SNP.apply(list))
			print(len(list(gene_to_snp.keys())))
			
			memento.compute_2d_moments(adata, list(gene_to_snp.keys()))

			memento.ht_2d_moments(
				adata, 
				covariate=cov_df,
				treatment=donor_df,
				treatment_for_gene=gene_to_snp,
				num_boot=5000, 
				verbose=1,
				num_cpus=80,
				resampling='bootstrap',
				approx=True,
				resample_rep=True)

		#     adata.write(data_path + 'memento_1k/{}.h5ad'.format(ct))
			memento_result = memento.get_2d_ht_result(adata)
			memento_result.to_csv(data_path + 'memento_full/{}/{}_{}_block_{}_coexpression.csv'.format(window,pop, ct, block), index=False)