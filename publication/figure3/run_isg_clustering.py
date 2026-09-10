"""Rebuild the canonical / non-canonical ISG classification that panels D-G rest on.

Port of publication/original/hbec_interferon/classify_isg/select_isgs.ipynb cells 30-52.

The original wrote its gene lists to `canonical_isgs.pkl` / `noncanonical_isgs.pkl`,
which are gone, so they are recomputed here: gene-by-gene memento correlations over
ciliated cells in control and under IFN-beta at 6h, then agglomerative clustering.

This script does NOT define the gene lists -- `isg_gene_lists.py` recovers those from
Supplementary Table 2 and the notebook's stored output. What this does is check how far
the clustering still reproduces, which is worth knowing and was worth measuring:

  * The canonical module survives. Clustering the control correlations at the notebook's
    distance_threshold=15 puts all 72 published canonical genes in one cluster, plus 11
    extra interferon genes (Jaccard 0.87).
  * The non-canonical modules do not. One recovered cluster matches the notebook's
    cluster 2 on every summary statistic (61 genes, mean within-cluster correlation
    0.430 vs 0.428) while sharing only 13 of its 61 genes -- the partition is degenerate
    at that level, so small differences in the correlation estimates reshuffle it.

Hence the published table, not this clustering, is the source of truth.

    python run_isg_clustering.py correlations   # gene-pair moments, writes csv
    python run_isg_clustering.py validate       # report agreement with the published lists
"""

import argparse
import itertools
import os

import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.cluster import AgglomerativeClustering

import config
import isg_gene_lists
import memento_legacy

config.add_repo_to_path()
import memento  # noqa: E402

COUNTS = config.FIGURE3_DATA + 'panelDEFG_isg/HBEC_type_I_filtered_counts_deep.h5ad'
CILIATED = 'C'
STIM_TIMEPOINT = '6'

# Markers the notebook used to locate its clusters.
# Cell 38 of the notebook identified the canonical module by MX1's label. Cell 37
# shows STAT2 landing in a different cluster, so it is not a co-anchor.
CANONICAL_ANCHOR = 'MX1'


def load_ciliated():
    adata = sc.read(COUNTS)
    adata.obs['ct'] = config.abbreviate_cell_types(adata.obs['cell_type'])
    adata.obs['q'] = adata.obs['batch'].apply(config.assign_capture_efficiency)
    memento.setup_memento(adata, q_column='q', trim_percent=0.1)
    return adata[adata.obs['ct'] == CILIATED].copy()


def type1_response_genes():
    """Genes induced by IFN-beta at 6h, the gene universe the clustering runs over."""
    table = memento_legacy.read_1d_ht(
        config.FIGURE3_DATA + f'panelDEFG_isg/tests/{CILIATED}_beta_{STIM_TIMEPOINT}.h5ad')
    return table.query('de_coef > 0.5 & de_fdr < 0.05')['gene'].tolist()


def gene_by_gene(adata, genes):
    """Symmetric memento correlation matrix over `genes`."""
    memento.compute_2d_moments(adata, list(itertools.combinations(genes, 2)))
    moments = memento.get_2d_moments(adata, groupby='group')

    matrix = pd.DataFrame(0.0, index=genes, columns=genes)
    for _, row in moments.iterrows():
        matrix.loc[row['gene_1'], row['gene_2']] = row['group_1']
        matrix.loc[row['gene_2'], row['gene_1']] = row['group_1']
    shared = [g for g in matrix.index if g in matrix.columns]
    return matrix.loc[shared, shared]


def _prepare(adata, subset_mask):
    subset = adata[subset_mask].copy()
    subset.obs['group'] = 1
    memento.create_groups(subset, label_columns=['donor', 'group'])
    memento.compute_1d_moments(subset, min_perc_group=0.9)
    return subset


def compute_correlations():
    adata = load_ciliated()
    genes = type1_response_genes()
    print(f'{adata.shape[0]} ciliated cells, {len(genes)} IFN-beta induced genes', flush=True)

    control = _prepare(adata, adata.obs['stim'] == 'control')
    stim = _prepare(adata, (adata.obs['stim'] == 'beta')
                    & (adata.obs['time'] == STIM_TIMEPOINT))

    shared = sorted(set(control.var.index) & set(stim.var.index) & set(genes))
    print(f'{len(shared)} genes survive the moment filters in both conditions', flush=True)

    for name, subset in [('control', control), ('stim', stim)]:
        matrix = gene_by_gene(subset, shared)
        matrix.to_csv(config.intermediate_path(f'gxg_{name}.csv'))
        print(f'  wrote gxg_{name}.csv {matrix.shape}', flush=True)

    # Per-timepoint IFN-beta correlations, for the panel D network.
    for timepoint in config.TIMEPOINTS:
        subset = _prepare(adata, (adata.obs['stim'] == 'beta')
                          & (adata.obs['time'] == timepoint))
        present = [g for g in shared if g in subset.var.index]
        matrix = gene_by_gene(subset, present)
        matrix.to_csv(config.intermediate_path(f'gxg_beta_{timepoint}.csv'))
        print(f'  wrote gxg_beta_{timepoint}.csv {matrix.shape}', flush=True)


def cluster_matrix(matrix, distance_threshold=15):
    clipped = matrix.clip(upper=1, lower=-1)
    return AgglomerativeClustering(
        n_clusters=None, distance_threshold=distance_threshold).fit(clipped)


def describe_clusters(labels, stim_gxg, ctrl_gxg):
    """Size and mean within-cluster correlation, the diagnostic the notebook printed."""
    rows = []
    for cluster in range(labels.max() + 1):
        members = np.where(labels == cluster)[0]
        rows.append({
            'cluster': cluster,
            'size': members.shape[0],
            'mean_stim_corr': stim_gxg.iloc[members, members].values.mean(),
            'mean_ctrl_corr': ctrl_gxg.iloc[members, members].values.mean(),
            'genes': ','.join(stim_gxg.index[members][:6]),
        })
    return pd.DataFrame(rows)


def validate_clustering(distance_threshold):
    """Report how far the recomputed clustering agrees with the published gene lists."""
    ctrl_gxg = pd.read_csv(config.intermediate_path('gxg_control.csv'), index_col=0)
    stim_gxg = pd.read_csv(config.intermediate_path('gxg_stim.csv'), index_col=0)

    published = isg_gene_lists.load()
    canonical = set(published.query('isg_class == "canonical"')['gene'])
    noncanonical = set(published.query('isg_class == "noncanonical"')['gene'])

    ctrl_labels = cluster_matrix(ctrl_gxg, distance_threshold).labels_
    print('control clustering:')
    print(describe_clusters(ctrl_labels, stim_gxg, ctrl_gxg).to_string(index=False))

    anchor = ctrl_labels[ctrl_gxg.index.get_loc(CANONICAL_ANCHOR)]
    recovered = set(ctrl_gxg.index[ctrl_labels == anchor])
    present = canonical & set(ctrl_gxg.index)
    print(f'\ncanonical module (cluster containing {CANONICAL_ANCHOR}): {len(recovered)} genes')
    print(f'  published canonical present in the matrix: {len(present)}')
    print(f'  recovered: {len(recovered & present)}  missed: {len(present - recovered)}  '
          f'extra: {len(recovered - present)}')
    print(f'  Jaccard: {len(recovered & present) / len(recovered | present):.3f}')
    if recovered - present:
        print(f'  extra genes: {", ".join(sorted(recovered - present))}')

    remaining = [g for g in stim_gxg.index if g not in canonical]
    subset_stim = stim_gxg.loc[remaining, remaining]
    subset_ctrl = ctrl_gxg.loc[remaining, remaining]
    stim_labels = cluster_matrix(subset_stim, distance_threshold).labels_

    summary = describe_clusters(stim_labels, subset_stim, subset_ctrl)
    overlaps = []
    for cluster in summary['cluster']:
        members = set(subset_stim.index[stim_labels == cluster])
        overlaps.append(len(members & noncanonical))
    summary['overlap_published_nc'] = overlaps
    print('\nstim clustering with the canonical module removed:')
    print(summary.drop(columns='genes').to_string(index=False))

    best = summary.loc[summary['overlap_published_nc'].idxmax()]
    print(f'\nbest-matching cluster recovers {int(best["overlap_published_nc"])} of '
          f'{len(noncanonical)} published non-canonical genes')
    print('The non-canonical partition does not reproduce; isg_gene_lists.py is '
          'the source of truth for the panels.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['correlations', 'validate'])
    parser.add_argument('--distance-threshold', type=float, default=15,
                        help="the notebook's working value")
    args = parser.parse_args()

    if args.command == 'correlations':
        compute_correlations()
    else:
        validate_clustering(args.distance_threshold)


if __name__ == '__main__':
    main()
