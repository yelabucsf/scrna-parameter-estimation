"""
	interferon_data_analysis.py
	This script executes most of the analysis for the inferferon data used in demuxlet. 
	Some of the preprocessing can be found in the interferon_preprocess.ipynb notebook.
"""


import pandas as pd
import matplotlib.pyplot as plt
import scanpy.api as sc
import scipy as sp
import numpy as np
import itertools
import scipy.stats as stats
import imp
import sc_estimator


if __name__ == '__main__':

	RUN_1D = False
	RUN_2D = True

	data_path = '/netapp/home/mincheol/param_est_data/interferon_data/'

	cd4_adata = sc.read(data_path + 'interferon.cd4.h5ad')
	cd4_adata_norm = sc.read(data_path + 'interferon.cd4.norm.h5ad')

	isg_candidates = pd.read_csv(data_path + 'isg_candidates.csv', index_col=0)
	isg_genes = isg_candidates['Gene Name'].tolist()
	irf_genes = [gene for gene in cd4_adata.var.index if gene[:3] == 'IRF']

	estimator = sc_estimator.SingleCellEstimator(cd4_adata, group_label='stim')

	if RUN_1D:
		estimator = sc_estimator.SingleCellEstimator(cd4_adata, group_label='stim')
		for idx, gene in enumerate(isg_genes + irf_genes):

			if gene not in cd4_adata.var.index:
				continue

			estimator.compute_1d_params(gene, group='stim', initial_sigma_hat = 5, initial_mu_hat=5, num_iter=200)
			estimator.compute_1d_params(gene, group='ctrl', initial_sigma_hat = 5, initial_mu_hat=5, num_iter=200)
			estimator.differential_expression(gene, groups=('ctrl', 'stim'))

			estimator.export_model(data_path + 'model_save/param_1d_{}.pkl'.format(idx))

	if RUN_2D:

		estimator = sc_estimator.SingleCellEstimator(cd4_adata, group_label='stim')
		estimator.import_model(data_path + 'model_save/param_1d_{}.pkl'.format(61))

		param_2d_counter = 0
		for irf_gene, isg_gene in itertools.product(['IRF7'], isg_genes):

			estimator.compute_2d_params(irf_gene, isg_gene, group='ctrl', search_num=50)
			estimator.compute_2d_params(irf_gene, isg_gene, group='stim', search_num=50)

			estimator.export_model(data_path + 'model_save/param_2d_{}_50_IRF7.pkl'.format(param_2d_counter))

			param_2d_counter += 1

		#estimator.export_model(data_path + 'model_save/param_2d_finished.pkl')

	print('Completed run!')