"""
	Run PCA using the covariance matrix estimated with empirical Bayes
"""

import numpy as np
import scanpy.api as sc
import simplesc


if __name__ == '__main__':

	data_path = '/netapp/home/mincheol/parameter_estimation/inteferon_data/'

	adata = sc.read(data_path + 'interferon.raw.h5ad')

	estimator = simplesc.SingleCellEstimator(
	    adata=adata, 
	    group_label='cell',
	    n_umis_column='n_counts',
	    num_permute=10000,
	    p=0.1)

	x_pca = estimator.pca()

	np.save(data_path + 'x_pca_all.npy', x_pca)

