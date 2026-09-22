import memento
import pandas as pd


def _check_genetics_input(inds_list, snps, cov):
	"""
	Check the inputs for genetics analysis.
	"""
	
	assert set(inds_list) == set(snps.index)
	assert set(inds_list) == set(cov.index)

	
def run_eqtl(
	adata, 
	snps, 
	cov, 
	gene_snp_pairs, 
	num_cpu, 
	donor_column, 
	num_blocks=5,
	mean_expr_threshold=0.01,
	num_boot=5000):
	"""
	Run the legacy CPU mean-only eQTL workflow with replicate resampling.

	``snps`` and ``cov`` are donor-indexed numeric DataFrames. ``gene_snp_pairs``
	contains columns named ``gene`` and ``SNP``. ``donor_column`` names the
	observation column identifying donors. This wrapper does not expose a GPU
	backend; use ht_1d_moments with gene-specific dictionaries for GPU analysis.
	"""

	# Check genetics input
	_check_genetics_input(adata.obs[donor_column].drop_duplicates().tolist(), snps, cov)

	# Setup memento
	adata = adata.copy().copy()
	adata.obs['capture_rate'] = 0.1
	memento.setup_memento(adata, q_column ='capture_rate', trim_percent=0.1, filter_mean_thresh=mean_expr_threshold, estimator_type='mean_only')
	memento.create_groups(adata, label_columns=[donor_column])

	# To prevent memory issues, we will separate out some SNPs for each gene
	reordered_gene_snp_pairs = gene_snp_pairs.sample(frac=1)

	# Re-order the SNP and covariate DataFrames to the order that memento expects
	sample_order = memento.get_groups(adata) #the order of samples that memento expects
	cov_df = cov.loc[sample_order[donor_column]]
	snps_df = snps.loc[sample_order[donor_column]]

	# Compute 1D moments
	memento.compute_1d_moments(adata, min_perc_group=.3, gene_list=reordered_gene_snp_pairs.gene.drop_duplicates().tolist())

	# Number of tests in each block
	blocksize=int(gene_snp_pairs.shape[0]/num_blocks)
	print('blocksize', blocksize)

	# Run memento for each block
	results = []
	for block in range(num_blocks):
		print('working on block', block)
		subset_gene_snp_pairs = reordered_gene_snp_pairs.iloc[(blocksize*block):(blocksize*(block+1)), :]

		gene_to_snp_dictionary = dict(subset_gene_snp_pairs[subset_gene_snp_pairs.gene.isin(adata.var.index)].groupby('gene')['SNP'].apply(list))

		print(len(gene_to_snp_dictionary.keys()))

		memento.ht_1d_moments(
			adata, 
			covariate=cov_df,
			treatment=snps_df,
			treatment_for_gene=gene_to_snp_dictionary,
			num_boot=num_boot,
			verbose=1,
			num_cpus=num_cpu,
			approx='norm',
			resample_rep=True)
		results.append(memento.get_1d_ht_result(adata))
	return pd.concat(results)[['gene', 'tx', 'de_coef', 'de_se', 'de_pval']]


def binary_test_1d(adata, capture_rate, treatment_col, num_cpus, num_boot=5000, verbose=1, replicates=[]):
	"""
	Compare mean and variability for a binary treatment using the CPU backend.

	``replicates`` is a list of observation column names added to the grouping
	alongside ``treatment_col``. It does not add regression covariates, account
	for donor pairing, or enable replicate resampling. Use the explicit setup,
	grouping, moment, and testing functions for those designs or GPU execution.
	Returns a result DataFrame with unadjusted p-values; the input is copied.
	"""
	
	adata = adata.copy().copy()
	adata.obs['capture_rate'] = capture_rate
	memento.setup_memento(adata, q_column='capture_rate')
	memento.create_groups(adata, label_columns=[treatment_col]+replicates)
	memento.compute_1d_moments(adata, min_perc_group=.9)
	sample_meta = memento.get_groups(adata)[[treatment_col]]
	memento.ht_1d_moments(
		adata, 
		treatment=sample_meta,
		num_boot=num_boot, 
		verbose=verbose,
		num_cpus=num_cpus)
	result_1d = memento.get_1d_ht_result(adata)
	return result_1d


def binary_test_2d(adata, gene_pairs, capture_rate, treatment_col, num_cpus, num_boot=5000, verbose=1, replicates=[]):
	"""
	Compare correlation for a binary treatment using the CPU backend.

	``gene_pairs`` contains (gene_1, gene_2) tuples. Pairs with genes removed by
	filtering are omitted. ``replicates`` adds observation column names to the
	grouping only: it does not add donor covariates, account for pairing, or
	enable replicate resampling. Use the explicit workflow for those controls
	or GPU execution. Returns a result DataFrame; the input is copied.
	"""
	
	adata = adata.copy().copy()
	adata.obs['capture_rate'] = capture_rate
	memento.setup_memento(adata, q_column='capture_rate')
	memento.create_groups(adata, label_columns=[treatment_col]+replicates)
	memento.compute_1d_moments(adata, min_perc_group=.9)
	filtered_gene_pairs = [(a,b) for a,b in gene_pairs if a in adata.var.index and b in adata.var.index]
	memento.compute_2d_moments(adata, filtered_gene_pairs)
	sample_meta = memento.get_groups(adata)[[treatment_col]]
	memento.ht_2d_moments(
		adata, 
		treatment=sample_meta,
		num_boot=num_boot, 
		verbose=verbose,
		num_cpus=num_cpus)
	result_2d = memento.get_2d_ht_result(adata)
	return result_2d
