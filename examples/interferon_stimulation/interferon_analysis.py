"""
	interferon_analysis.py

	This script contains the code to fit all of the parameters and confidence intervals for
	the interferon analysis with scme.
"""

import scanpy.api as sc
import time
import pickle as pkl
import sys
import os
#sys.path.append('/wynton/group/ye/mincheol/Github/scrna-parameter-estimation/simplesc')
sys.path.append('/home/mkim7/Github/scrna-parameter-estimation/scmemo')
import scmemo, utils

# Global variable with the data path relevant for this analysis
data_path = '/data/parameter_estimation/interferon_data/20200324/'
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


def stim_effect_1d(adata):
	"""
		IFN-beta stimulation effect on PBMCs.
	"""

	estimator = scmemo.SingleCellEstimator(
		adata=adata, 
		group_label='cell_type',
		n_umis_column='n_counts',
		num_permute=200000,
		beta=0.1)

	estimator.estimate_beta_sq(tolerance=3)
	estimator.estimate_parameters()

	estimator.compute_confidence_intervals_1d(
		groups=estimator.groups,
		groups_to_compare=[(ct + ' - ctrl', ct + ' - stim') for ct in adata.obs['cell'].drop_duplicates()])
	
	# Save the 1D hypothesis test result
	with open(data_path + 'stim_effect_1d.pkl', 'wb') as f:
		pkl.dump(estimator.hypothesis_test_result_1d, f)

	# Save the 1d parameters
	with open(data_path + 'stim_effect_1d_params.pkl', 'wb') as f:
		param_dict = {group:{k:v for k, v in estimator.parameters[group].items() if k != 'corr'} for group in estimator.parameters}
		pkl.dump(param_dict, f)

	# Save the central moments	
	with open(data_path + 'stim_effect_1d_moments.pkl', 'wb') as f:
		moment_dict = {group:{k:v for k, v in estimator.estimated_central_moments[group].items() if k != 'prod'} for group in estimator.estimated_central_moments}
		pkl.dump(moment_dict, f)

	# Save the 1d parameters
	with open(data_path + 'stim_effect_ci_1d.pkl', 'wb') as f:
		pkl.dump(estimator.parameters_confidence_intervals, f)


def stim_effect_2d(adata, gene_list):
	"""
		IFN-beta stimulation effect on PBMCs.
	"""

	estimator = scmemo.SingleCellEstimator(
		adata=adata, 
		group_label='cell_type',
		n_umis_column='n_counts',
		num_permute=10000,
		beta=0.1)

	estimator.estimate_beta_sq(tolerance=3)
	estimator.estimate_parameters()

	with open(data_path + 'immune_genes.pkl', 'rb') as f:
		immune_genes_to_test = pkl.load(f)
	print(len(immune_genes_to_test))

	# with open(data_path + 'tfs_to_consider.pkl', 'rb') as f:
	#     tfs_in_highcount = pkl.load(f)

	# tfs = ['JUN',
	# 	'ATF3',
	# 	'STAT1',
	# 	'STAT4',
	# 	'FOXP1',
	# 	'ATF6B',
	# 	'ATF1',
	# 	'STAT2',
	# 	'STAT6',
	# 	'FOS',
	# 	'BATF',
	# 	'AATF',
	# 	'STAT3',
	# 	'JUNB',
	# 	'JUND',
	# 	'ATF5',
	# 	'ATF4']

	tfs = ['RAD21',
		'CEBPB',
		'SMARCB1',
		'CEBPZ',
		'H2AFZ',
		'IRF1',
		'ETS1', 
		'REST',
		'BACH1',
		'NELFE',
		'BDP1',
		'YY1',
		'MEF2A',
		'IRF3',
		'HMGN3',
		'BHLHE40',
		'GATA3',
		'SMARCC1',
		'MAX',
		'SMC3']

	for ct in adata.obs['cell'].drop_duplicates().tolist():

		print('Correlation testing for', ct)
		start = time.time()
		estimator.compute_confidence_intervals_2d(
			gene_list_1=tfs,
			gene_list_2=immune_genes_to_test,
			groups=[ct + ' - ctrl', ct + ' - stim'],
			groups_to_compare=[(ct + ' - ctrl', ct + ' - stim')])
		print('This cell type took', time.time()-start)

		with open(data_path + 'stim_effect_2d_random_{}.pkl'.format(ct), 'wb') as f:
			pkl.dump(estimator.hypothesis_test_result_2d[(ct + ' - ctrl', ct + ' - stim')], f)

	print('Saving 2D comparison results...')

	# Save the 2D hypothesis test result
	with open(data_path + 'stim_effect_2d_parameters_random.pkl', 'wb') as f:
		pkl.dump(estimator.parameters, f)


if __name__ == '__main__':

	# Read the AnnData object and filter out hemoglobin genes
	adata = sc.read(data_path + 'interferon_highcount.raw.h5ad')
	adata = adata[:, adata.var.index.map(lambda x: x[:2] != 'HB')].copy()
	adata.obs['cell_type'] = (adata.obs['cell'].astype(str) + ' - ' + adata.obs['stim'].astype(str)).astype('category')

	gene_list = adata.var.index.tolist()

	#stim_effect_1d(adata)

	stim_effect_2d(adata, gene_list)

