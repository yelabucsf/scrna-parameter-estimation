def get_differential_genes(
	gene_list,
	hypothesis_test_dict, 
	group_1, 
	group_2, 
	which, 
	direction, 
	sig=0.1, 
	num_genes=50):
	"""
		Get genes that are increased in expression in group 2 compared to group 1, sorted in order of significance.
		:which: should be either "mean" or "dispersion"
		:direction: should be either "increase" or "decrease"
		:sig: defines the threshold
		:num_genes: defines the number of genes to be returned. If bigger than the number of significant genes, then return only the significant ones.
	"""

	# Setup keys
	group_key = (group_1, group_2)
	param_key = 'de' if which == 'mean' else 'dv'

	# Find the number of genes to return
	sig_condition = hypothesis_test_dict[group_key][param_key + '_fdr'] < sig
	dir_condition = ((1 if direction == 'increase' else -1)*hypothesis_test_dict[group_key][param_key + '_diff']) > 0
	num_sig_genes = (sig_condition & dir_condition).sum()
	
	# We will order the output by significance, then by effect size. Just turn the FDR of the other half into 1's to remove them from the ordering.
	relevant_fdr = hypothesis_test_dict[group_key][param_key + '_fdr'].copy()
	relevant_fdr[~dir_condition] = 1
	relevant_effect_size = hypothesis_test_dict[group_key][param_key + '_diff'].copy()
	relevant_effect_size[~dir_condition] = 0

	# Get the order of the genes in terms of FDR.
	df = pd.DataFrame()
	df['pval'] = hypothesis_test_dict[group_key][param_key + '_pval'].copy()
	df['fdr'] = relevant_fdr
	df['effect_size'] = np.absolute(relevant_effect_size)
	df['gene'] = gene_list
	df = df.sort_values(by=['fdr', 'effect_size'], ascending=[True, False])
	df['rank'] = np.arange(df.shape[0])+1

	df = df.query('fdr < {}'.format(sig)).iloc[:num_genes, :].copy()

	return df


	return relevant_fdr[order], hypothesis_test_dict[group_key][param_key + '_diff'][order], gene_list[order], order
