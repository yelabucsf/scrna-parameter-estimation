"""
	cd4_analysis.py

	This script contains the code to fit all of the parameters and confidence intervals for
	the CD4 cropseq data with scme.
"""

import scanpy.api as sc
import time
import scme
import pickle as pkl
import sys
import os

# Global variable with the data path relevant for this analysis
data_path = '/wynton/group/ye/mincheol/parameter_estimation/cd4_cropseq_data/'


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


def corr_ko_effect(adata):
    """
        IFN-beta stimulation effect on PBMCs.
    """

    estimator = scme.SingleCellEstimator(
        adata=adata, 
        group_label='group',
        n_umis_column='n_counts',
        num_permute=10000,
        beta=0.1)

    print('Estimating beta sq')
    estimator.estimate_beta_sq(tolerance=3)
    estimator.estimate_parameters()

    with open(data_path + 'genes_to_test.pkl', 'rb') as f:
        genes_to_test = pkl.load(f)

    with open(data_path + '../interferon_data/immune_genes.pkl', 'rb') as f:
        immune_genes = pkl.load(f)

    for group in adata.obs['group'].drop_duplicates():

        if group == 'WT':
            continue

        if group.split('.')[0] not in genes_to_test:
            continue

        print('Correlation testing for', group)
        start = time.time()
        estimator.compute_confidence_intervals_2d(
            gene_list_1=genes_to_test,
            gene_list_2=immune_genes,
            groups=['WT', group],
            groups_to_compare=[('WT', group)])
        print('This cell type took', time.time()-start)

        with open(data_path + 'diff_cor/{}.pkl'.format(group), 'wb') as f:
            pkl.dump(estimator.hypothesis_test_result_2d, f)

        idxs_1 = estimator.hypothesis_test_result_2d[('WT', group)]['gene_idx_1']
        idxs_2 = estimator.hypothesis_test_result_2d[('WT', group)]['gene_idx_2']

        with open(data_path + 'ko_ci/{}.pkl'.format(group), 'wb') as f:
            ci_dict = {}
            for group, val in estimator.parameters_confidence_intervals.items():
                ci_dict[group] = {'corr':val['corr'][idxs_1, :][:, idxs_2]}
            pkl.dump(ci_dict, f)


if __name__ == '__main__':

    # Read the AnnData object and filter out hemoglobin genes
    adata = sc.read(data_path + 'guide_singlets.h5ad')

    corr_ko_effect(adata)
