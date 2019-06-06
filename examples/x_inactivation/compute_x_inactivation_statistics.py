"""
	compute_x_intactivation_statistics.py
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
import scanpy.api as sc
import scipy as sp
import itertools
import numpy as np
import scipy.stats as stats
import scipy.sparse as sparse
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')


if __name__ == '__main__':

	data_path = '/netapp/home/mincheol/parameter_estimation/x_inactivation_data/'

	full_adata = sc.read(data_path + 'lupus_annotated_nonorm_V6_x_genes.h5ad')

	if len(sys.argv) > 1 and sys.argv[1] == 'ct': # Compute CT differences for a single individual

		import simplesc

		sampled_ind_list = pd.read_csv(data_path + 'lupus_sampled_ind_list.csv')['ind_cov'].tolist()

		adata = full_adata[(full_adata.obs.ct_cov == ct) & (full_adata.obs.ind_cov.isin(sampled_ind_list))].copy()

		estimator = simplesc.SingleCellEstimator(
			adata,
			num_permute=25,
			p=0.1, group_label='ct_cov')

		for ct in adata.obs.ct_cov.drop_duplicates():

			print(ct)

			# Compute the CT stats
			estimator.compute_observed_statistics(group=ct)
			estimator.compute_params(group=ct)
			estimator.compute_permuted_statistics(group=ct)

			# Compute the rest of the cells
			estimator.compute_observed_statistics(group=('-' + ct))
			estimator.compute_params(group=('-' + ct))
			estimator.compute_permuted_statistics(group=('-' + ct))

			# Perform differential expression
			de_t_stats, de_null_stats, de_pvals = \
				estimator.differential_expression(ct, '-' + ct, method='perm')

			# Perform differential variance
			dv_t_stats, dv_null_stats, dv_pvals = \
				estimator.differential_variance(ct, '-' + ct, method='perm')

			# Save DE result
			np.save(data_path + 'ct_statistics/{}_{}_de_t_stats.npy'.format(ind, ct), de_t_stats)
			np.save(data_path + 'ct_statistics/{}_{}_de_null_t_stats.npy'.format(ind, ct), de_null_stats)
			np.save(data_path + 'ct_statistics/{}_{}_de_pvals.npy'.format(ind, ct), de_pvals)

			# Save DV result
			np.save(data_path + 'ct_statistics/{}_{}_dv_t_stats.npy'.format(ind, ct), dv_t_stats)
			np.save(data_path + 'ct_statistics/{}_{}_dv_null_t_stats.npy'.format(ind, ct), dv_null_stats)
			np.save(data_path + 'ct_statistics/{}_{}_dv_pvals.npy'.format(ind, ct), dv_pvals)

	elif len(sys.argv) > 1 and sys.argv[1] == 'sex': # CT specific sex differences

		import simplesc

		sampled_ind_list = pd.read_csv(data_path + 'lupus_sampled_ind_list.csv')['ind_cov'].tolist()

		ct = pd.read_csv(data_path + 'lupus_ct_list.csv')['ct_cov'].tolist()[int(sys.argv[2])]

		# Filter the full AnnData object
		adata = full_adata[(full_adata.obs.ct_cov == ct) & (full_adata.obs.ind_cov.isin(sampled_ind_list))].copy()
		adata.obs['Female'] = adata.obs['Female'].astype(int).astype(str)

		# Instantiate the estimator
		estimator = simplesc.SingleCellEstimator(
			adata,
			num_permute=50,
			p=0.1, group_label='Female')

		# Compute the CT stats
		estimator.compute_observed_statistics(group='0')
		estimator.compute_params(group='0')
		estimator.compute_permuted_statistics(group='0')

		# Compute the rest of the cells
		estimator.compute_observed_statistics(group='1')
		estimator.compute_params(group='1')
		estimator.compute_permuted_statistics(group='1')

		# Perform differential expression
		de_t_stats, de_null_stats, de_pvals = \
			estimator.differential_expression('1', '0', method='perm')

		# Perform differential variance
		dv_t_stats, dv_null_stats, dv_pvals = \
			estimator.differential_variance('1', '0', method='perm')

		# Save DE result
		np.save(data_path + 'sex_statistics/{}_de_t_stats.npy'.format(ct), de_t_stats)
		np.save(data_path + 'sex_statistics/{}_de_null_t_stats.npy'.format(ct), de_null_stats)
		np.save(data_path + 'sex_statistics/{}_de_pvals.npy'.format(ct), de_pvals)

		# Save DV result
		np.save(data_path + 'sex_statistics/{}_dv_t_stats.npy'.format(ct), dv_t_stats)
		np.save(data_path + 'sex_statistics/{}_dv_null_t_stats.npy'.format(ct), dv_null_stats)
		np.save(data_path + 'sex_statistics/{}_dv_pvals.npy'.format(ct), dv_pvals)

	else: 

		# Compile all CT difference t-statistics into a CSV.

		ind_list = pd.read_csv(data_path + 'lupus_ind_list.csv')['ind_cov'].tolist()

		for ct in full_adata.obs.ct_cov.drop_duplicates():

			de_stats = []
			de_pvals = []
			dv_stats = []
			dv_pvals = []
			ct_ind_list = []

			for ind in ind_list:

				try:
					de_stats.append(np.load(data_path + 'ct_statistics/{}_{}_de_t_stats.npy'.format(ind, ct)))
					de_pvals.append(np.load(data_path + 'ct_statistics/{}_{}_de_pvals.npy'.format(ind, ct)))
					dv_stats.append(np.load(data_path + 'ct_statistics/{}_{}_dv_t_stats.npy'.format(ind, ct)))
					dv_pvals.append(np.load(data_path + 'ct_statistics/{}_{}_dv_pvals.npy'.format(ind, ct)))
					ct_ind_list.append(ind)
				except:
					print('{} person is missing in {} cell type!'.format(ind, ct))
					continue

			pd.DataFrame(
				data=np.vstack(de_stats),
				index=ct_ind_list,
				columns=full_adata.var.index.values).to_csv(data_path + 'ct_combined_statistics/{}_de_t_stats.csv'.format(ct))
			pd.DataFrame(
				data=np.vstack(de_pvals),
				index=ct_ind_list,
				columns=full_adata.var.index.values).to_csv(data_path + 'ct_combined_statistics/{}_de_pvals.csv'.format(ct))
			pd.DataFrame(
				data=np.vstack(dv_stats),
				index=ct_ind_list,
				columns=full_adata.var.index.values).to_csv(data_path + 'ct_combined_statistics/{}_dv_t_stats.csv'.format(ct))
			pd.DataFrame(
				data=np.vstack(dv_pvals),
				index=ct_ind_list,
				columns=full_adata.var.index.values).to_csv(data_path + 'ct_combined_statistics/{}_dv_pvals.csv'.format(ct))
