"""
	cd4_analysis.py

	This script contains the code to fit all of the parameters and confidence intervals for
	the CD4 cropseq data with scme.
"""

import scanpy.api as sc
import time
import pickle as pkl
import os
import sys
sys.path.append('/home/mkim7/Github/scrna-parameter-estimation/simplesc')
import scme, utils
import glob

# Global variable with the data path relevant for this analysis
data_path = '/data/parameter_estimation/cd4_cropseq_data/'

import warnings
warnings.filterwarnings('ignore')


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


def mv_ko_effect(adata):
    """
        CRISPR KO effect on mean and variance of gene expression in PBMCs.
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

    # Save the central moments  
    with open(data_path + 'ko_effect_1d_moments.pkl', 'wb') as f:
        moment_dict = {group:{k:v for k, v in estimator.estimated_central_moments[group].items() if k != 'prod'} for group in estimator.estimated_central_moments}
        pkl.dump(moment_dict, f)


def corr_ko_effect(adata):
    """
        CRISPR KO effect on correlation of gene expression in PBMCs.
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

    with open(data_path + 'ko_genes_to_test.pkl', 'rb') as f:
        ko_genes_to_test = pkl.load(f)

    with open(data_path + '../interferon_data/immune_genes_to_test.pkl', 'rb') as f:
        immune_genes_to_test = pkl.load(f)

    for group in adata.obs['group'].drop_duplicates():

        print(group)

        if group == 'WT':
            continue

        if group.split('.')[0] not in ko_genes_to_test:
            print('not in KO genes to test')
            continue

        print('Correlation testing for', group)
        estimator.compute_confidence_intervals_2d(
            gene_list_1=ko_genes_to_test,
            gene_list_2=immune_genes_to_test,
            groups=['WT', group],
            groups_to_compare=[('WT', group)])

        with open(data_path + 'diff_cor/{}.pkl'.format(group), 'wb') as f:
            pkl.dump(estimator.hypothesis_test_result_2d[('WT', group)], f)

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

    with open(data_path + 'ko_genes_to_test.pkl', 'rb') as f:
        ko_genes_to_test = pkl.load(f)

    adata.obs['target_regulator'] = adata.obs['group']\
        .apply(lambda x: x.split('.')[0])

    adata = adata[adata.obs['target_regulator'].isin(ko_genes_to_test) | (adata.obs['group'] == 'WT')].copy()

    corr_ko_effect(adata)

    mv_ko_effect(adata)
