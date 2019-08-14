"""
	interferon_analysis.py

	This script contains the code to fit all of the parameters and confidence intervals for
	the interferon analysis with scme.
"""

import scanpy.api as sc
import time
import scme
import pickle as pkl
import sys
import os

# Global variable with the data path relevant for this analysis
data_path = '/wynton/group/ye/mincheol/parameter_estimation/interferon_data/'


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


def stim_effect(adata):
	"""
		IFN-beta stimulation effect on PBMCs.
	"""

	estimator = scme.SingleCellEstimator(
		adata=adata, 
		group_label='cell_type',
		n_umis_column='n_counts',
		num_permute=10000,
		beta=0.1)

	estimator.estimate_beta_sq(tolerance=3)
	estimator.estimate_parameters()

	# estimator.compute_confidence_intervals_1d(
	# 	groups=estimator.groups,
	# 	groups_to_compare=[(ct + ' - ctrl', ct + ' - stim') for ct in adata.obs['cell'].drop_duplicates()])
	
	# # Save the 1D hypothesis test result
	# with open(data_path + 'stim_effect_1d.pkl', 'wb') as f:
	# 	pkl.dump(estimator.hypothesis_test_result_1d, f)

	for ct in adata.obs['cell'].drop_duplicates():

		print('Correlation testing for', ct)
		start = time.time()
		estimator.compute_confidence_intervals_2d(
			gene_list_1=['STAT3', 'STAT4', 'STAT6', 'IRF1', 'IRF8', 'IRF9'],
			gene_list_2=adata.var.index.tolist(),
			groups=[ct + ' - ctrl', ct + ' - stim'],
			groups_to_compare=[(ct + ' - ctrl', ct + ' - stim')])
		print('This cell type took', time.time()-start)

	print('Saving 2D comparison results...')
	# Save the 2D hypothesis test result
	with open(data_path + 'stim_effect_2d.pkl', 'wb') as f:
		pkl.dump(estimator.hypothesis_test_result_2d, f)


if __name__ == '__main__':

	# Read the AnnData object and filter out hemoglobin genes
	adata = sc.read(data_path + 'interferon.raw.h5ad')
	adata = adata[:, adata.var.index.map(lambda x: x[:2] != 'HB')].copy()
	adata.obs['cell_type'] = (adata.obs['cell'].astype(str) + ' - ' + adata.obs['stim'].astype(str)).astype('category')

	# Find cell type markers
	#cell_type_markers(adata)

	# Find the cell type specific effect of IFN-B stimulation
	stim_effect(adata)







