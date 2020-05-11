"""
	interferon_analysis.py

	This script contains the code to fit all of the parameters and confidence intervals for
	the interferon analysis with scme.
"""

import scanpy.api as sc
import numpy as np
import time
import pickle as pkl
import sys
import os
#sys.path.append('/wynton/group/ye/mincheol/Github/scrna-parameter-estimation/simplesc')
sys.path.append('/home/mkim7/Github/scrna-parameter-estimation/scmemo')
import scmemo, utils

# Global variable with the data path relevant for this analysis
data_path = '/data/parameter_estimation/interferon_data/20200412/'
#data_path = '/wynton/group/ye/mincheol/parameter_estimation/interferon_data/20191218/'



class ForceIOStream:
	def __init__(self, stream):
		self.stream = stream

	def write(self, data):
		self.stream.write(data)
		self.stream.flush()
		if not self.stream.isatty():
			os.fsync(self.stream.fileno())

	def __getattr__(self, attr):
		return getattr(self.stream, attr)


sys.stdout = ForceIOStream(sys.stdout)
sys.stderr = ForceIOStream(sys.stderr)


def cell_type_markers(adata):
	"""
		Using the control data, detect the cell type markers.
	"""

	adata_ctrl = adata[adata.obs.stim == 'ctrl'].copy()

	estimator = scme.SingleCellEstimator(
		adata=adata_ctrl, 
		group_label='cell',
		n_umis_column='n_counts',
		num_permute=10000,
		beta=0.1)

	estimator.estimate_beta_sq(tolerance=3)
	estimator.estimate_parameters()

	start_time = time.time()
	estimator.compute_confidence_intervals_1d(
		groups=None,
		groups_to_compare=[(ct, '-' + ct) for ct in estimator.groups])
	print('Computing 1D confidence intervals took', time.time()-start_time)

	# Save the 1D hypothesis test result
	with open(data_path + 'ct_marker_1d.pkl', 'wb') as f:
		pkl.dump(estimator.hypothesis_test_result_1d, f)


def stim_effect_1d(adata, gene_list):
	"""
		IFN-beta stimulation effect on PBMCs.
	"""
	cts = ['CD4 T cells',  'CD14+ Monocytes', 'FCGR3A+ Monocytes', 'NK cells','CD8 T cells', 'B cells']
	
	for ct in cts:
		estimator = scmemo.SingleCellEstimator(
			adata=adata,
			covariate_label='stim',
			replicate_label='ind',
			batch_label=None,
			subsection_label='cell',
			num_permute=10000,
			covariate_converter={'ctrl':0, 'stim':1},
			q=.07,
			use_hat_matrix=True)

		estimator.estimate_q_sq(verbose=True, k=5)
		estimator.q_sq = 0.00600778
		estimator.compute_observed_moments(gene_list)
		estimator.estimate_1d_parameters(gene_list)

		estimator.setup_hypothesis_testing([ct])
		estimator.compute_effect_sizes_1d(gene_list)
		estimator.compute_confidence_intervals_1d(gene_list, verbose=True)
	
		# Save the 1D hypothesis test result
		with open(data_path + 'ind_replicate_result/hypothesis_test_1d_{}.pkl'.format(ct), 'wb') as f:
			pkl.dump(estimator.hypothesis_test_result, f)

		# Save the 1d parameters
		with open(data_path + 'ind_replicate_result/parameters_1d_{}.pkl'.format(ct), 'wb') as f:
			pkl.dump(estimator.parameters, f)

		# Save the 1d CI
		with open(data_path + 'ind_replicate_result/confidence_intervals_1d_{}.pkl'.format(ct), 'wb') as f:
			pkl.dump(estimator.parameters_confidence_intervals, f)


def stim_effect_2d(adata, job_num, gene_list):
	"""
		IFN-beta stimulation effect on PBMCs.
	"""
	
	# Get the TFs for this job
	with open(data_path + 'all_highcount_tfs.pkl', 'rb') as f:
		tfs = pkl.load(f)
	num_tfs_per_job = 5
	tfs = tfs[(job_num*num_tfs_per_job):(job_num+1)*num_tfs_per_job]
	print(tfs)
	
	cts = ['CD4 T cells',  'CD14+ Monocytes', 'FCGR3A+ Monocytes', 'NK cells','CD8 T cells', 'B cells']

	for ct in cts:
		estimator = scmemo.SingleCellEstimator(
			adata=adata,
			covariate_label='stim',
			replicate_label='ind',
			batch_label=None,
			subsection_label='cell',
			num_permute=10000,
			covariate_converter={'ctrl':0, 'stim':1},
			q=.07,
			use_hat_matrix=True)

		estimator.estimate_q_sq(verbose=True, k=5)
		estimator.q_sq = 0.00600778
		estimator.compute_observed_moments(gene_list)
		estimator.estimate_1d_parameters(gene_list)

		estimator.setup_hypothesis_testing([ct])
		estimator.compute_effect_sizes_2d(tfs, gene_list)
		estimator.compute_confidence_intervals_2d(tfs, gene_list, verbose=True)
	
		# Save the 1D hypothesis test result
		with open(data_path + 'ind_replicate_result/hypothesis_test_2d_{}_{}.pkl'.format(ct, job_num), 'wb') as f:
			pkl.dump(estimator.hypothesis_test_result, f)

		# Save the 1d parameters
		with open(data_path + 'ind_replicate_result/parameters_2d_{}_{}.pkl'.format(ct, job_num), 'wb') as f:
			pkl.dump(estimator.parameters, f)

		# Save the 1d CI
		with open(data_path + 'ind_replicate_result/confidence_intervals_2d_{}_{}.pkl'.format(ct, job_num), 'wb') as f:
			pkl.dump(estimator.parameters_confidence_intervals, f)


if __name__ == '__main__':

	# Read the AnnData object and filter out hemoglobin genes
	adata = sc.read(data_path + 'interferon_highcount.raw.h5ad')
	adata.X = adata.X.astype(np.int64)

	# Grab the highcount genes
	gene_list = adata.var.index.tolist()

# 	stim_effect_1d(adata, gene_list)

	job_num = int(sys.argv[1])
	stim_effect_2d(adata, job_num, gene_list)

