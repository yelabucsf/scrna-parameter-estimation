import numpy as np
import scipy.io as spio
import pandas as pd

if __name__ == '__main__':

	gene_to_eid = pd.read_csv('/ye/yelabstore2/mincheol/data/scrna-parameter-estimation/landed/gene_to_id.csv')

	gene_list = pd.read_csv('/ye/yelabstore2/mincheol/data/scrna-parameter-estimation/landed/hg19/genes.tsv', sep='\t', header=None)
	gene_list.columns = ['eid', 'gene']

	# Retrive and filter mRNA matrix
	gene_indices = gene_list[gene_list.eid.isin(gene_to_eid.eid)].index.values
	gene_labels = gene_list[gene_list.eid.isin(gene_to_eid.eid)]\
		.merge(gene_to_eid, on='eid', how='left').gene_y.values
	mRNA_matrix = spio.mmread('/ye/yelabstore2/mincheol/data/scrna-parameter-estimation/landed/hg19/matrix.mtx').tocsr()
	filtered_mRNA_matrix = mRNA_matrix[gene_indices, :].toarray()
	df_mRNA = pd.DataFrame(filtered_mRNA_matrix.T, columns=gene_labels)

	# Retrive and filter ADT matrix
	adt_matrix = pd.read_csv('/ye/yelabstore2/mincheol/data/scrna-parameter-estimation/landed/CITEseqADT.csv')
	adt_genes = [x.split('_')[0].upper() for x in adt_matrix.iloc[:, 0]]
	df_adt = adt_matrix.iloc[:, 1:]\
		.copy()\
		.transpose()
	df_adt.columns = adt_genes
	df_adt = df_adt[gene_to_eid.gene]

	# Save mRNA and ADT dataframes
	df_mRNA.to_csv('/ye/yelabstore2/mincheol/data/scrna-parameter-estimation/raw/mrna.csv', index=False)
	df_adt.to_csv('/ye/yelabstore2/mincheol/data/scrna-parameter-estimation/raw/adt.csv', index=False)







	
