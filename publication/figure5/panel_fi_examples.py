"""Figure 5F-I - worked examples of a vQTL and a cQTL.

Port of publication/genetics/run_memento/analyze_variability.ipynb cells 26-31 (F, G) and
analyze_coexpression.ipynb cells 53-59 (H, I).

  F  HLA-C expression variability per individual, by genotype at chr6:31326612 (eur, ncM)
  G  the HLA-C distribution in one representative individual per genotype
  H  JUNB-LYZ correlation per individual, by genotype at chr12:69688073 (asian, cM)
  I  LYZ against JUNB across single cells, all donors behind one representative individual

Both examples recompute their moments from the single-cell data, since the panels need
per-individual estimates rather than the pooled QTL summary statistics.

Panel F plots the residual variance, element 1 of `get_1d_moments`. The notebook reached
for element 0 -- the mean -- while labelling the axis "Variability"; that is a slip, since
a vQTL panel is about variability.
"""

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns

import config

config.add_repo_to_path()
import memento  # noqa: E402

SINGLE_CELL = config.FIGURE5_DATA + 'panelFI_examples/single_cell/'
GENOTYPES = config.FIGURE5_DATA + 'genotypes/'
CAPTURE_RATE = 0.1

VQTL = {'pop': 'eur', 'ct': 'ncM', 'snp': '6:31326612', 'gene': 'HLA-C',
        'mean_thresh': 0.05, 'min_perc_group': 0.2}
CQTL = {'pop': 'asian', 'ct': 'cM', 'snp': '12:69688073',
        'gene_1': 'JUNB', 'gene_2': 'LYZ', 'mean_thresh': 0.01, 'min_perc_group': 0.5}
MIN_CELLS_PER_DONOR = 100


def prepare(pop, cell_type, genes, mean_thresh, min_perc_group, min_cells=MIN_CELLS_PER_DONOR):
    """Load one population/cell type, keep genotyped donors, and estimate moments."""
    genotypes = pd.read_csv(GENOTYPES + f'{pop}_genos.tsv', sep='\t', index_col=0)
    adata = sc.read(SINGLE_CELL + f'{pop}_{cell_type}.h5ad')

    counts = adata.obs['ind_cov'].value_counts()
    keep = (adata.obs['ind_cov'].isin(genotypes.columns)
            & adata.obs['ind_cov'].isin(counts[counts > min_cells].index))
    adata = adata[keep].copy()

    adata.obs['capture_rate'] = CAPTURE_RATE
    memento.setup_memento(adata, q_column='capture_rate', trim_percent=0.1,
                          filter_mean_thresh=mean_thresh)
    memento.create_groups(adata, label_columns=['ind_cov'])
    memento.compute_1d_moments(adata, min_perc_group=min_perc_group, gene_list=genes)
    return adata, genotypes


def _strip_group_prefix(frame):
    """memento group columns look like 'sg^<donor>'; keep the donor id."""
    frame = frame.set_index('gene')
    frame.columns = [column[3:] for column in frame.columns]
    return frame


def panel_fg(axes):
    adata, genotypes = prepare(VQTL['pop'], VQTL['ct'], [VQTL['gene']],
                               VQTL['mean_thresh'], VQTL['min_perc_group'])
    # get_1d_moments returns (mean, residual variance, cell counts).
    _, variability, _ = memento.get_1d_moments(adata)
    estimate = _strip_group_prefix(variability)

    info = pd.concat([genotypes.loc[VQTL['snp']], estimate.loc[VQTL['gene']]], axis=1).dropna()
    info.columns = [VQTL['snp'], 'variability']
    print(f'panel F: {info.shape[0]} individuals with genotype and estimate')
    print(info.groupby(VQTL['snp'])['variability'].agg(['count', 'mean']).round(3).to_string())

    sns.boxplot(x=VQTL['snp'], y='variability', data=info, color='gray',
                fliersize=0, ax=axes[0])
    sns.stripplot(x=VQTL['snp'], y='variability', data=info, color='black', s=3, ax=axes[0])
    axes[0].set_ylabel(f'{VQTL["gene"]} variability')
    axes[0].set_xlabel(f'Genotype at chr{VQTL["snp"]}')

    # Panel G: one representative individual per genotype, at the median estimate.
    donor_genotype = info[VQTL['snp']]
    for genotype in sorted(donor_genotype.unique()):
        candidates = info[info[VQTL['snp']] == genotype]['variability']
        representative = (candidates - candidates.median()).abs().idxmin()
        expression = adata[adata.obs['ind_cov'] == representative, VQTL['gene']].X.todense().A1
        axes[1].hist(np.log(expression + 1), bins=30, histtype='step', lw=1.5,
                     density=True, label=f'genotype {int(genotype)}')
    axes[1].set_xlabel(f'log({VQTL["gene"]} + 1)')
    axes[1].set_ylabel('Density')
    axes[1].legend(frameon=False)


def panel_hi(axes):
    adata, genotypes = prepare(CQTL['pop'], CQTL['ct'], [CQTL['gene_1'], CQTL['gene_2']],
                               CQTL['mean_thresh'], CQTL['min_perc_group'], min_cells=0)
    memento.compute_2d_moments(adata, [(CQTL['gene_1'], CQTL['gene_2'])])
    # Without `groupby`, get_2d_moments returns (correlations, cell_counts).
    correlations, _ = memento.get_2d_moments(adata)

    values = correlations.iloc[0].drop(['gene_1', 'gene_2'])
    values.index = [str(name)[3:] for name in values.index]
    info = pd.concat([genotypes.loc[CQTL['snp']], values.rename('coexpression')],
                     axis=1).dropna()
    info.columns = [CQTL['snp'], 'coexpression']
    info = info.query('-1 < coexpression < 1')
    print(f'panel H: {info.shape[0]} individuals with genotype and correlation')
    print(info.groupby(CQTL['snp'])['coexpression'].agg(['count', 'mean']).round(3).to_string())

    sns.boxplot(x=CQTL['snp'], y='coexpression', data=info, color='gray',
                fliersize=0, ax=axes[0])
    sns.stripplot(x=CQTL['snp'], y='coexpression', data=info, color='black', s=3, ax=axes[0])
    axes[0].set_ylabel(f'Correlation({CQTL["gene_1"]}, {CQTL["gene_2"]})')
    axes[0].set_xlabel(f'Genotype at chr{CQTL["snp"]}')

    # Panel I: all cells in grey, one representative individual over the top.
    def expression(mask, gene):
        return np.log(adata[mask, gene].X.todense().A1 + 1)

    everyone = np.ones(adata.shape[0], dtype=bool)
    axes[1].scatter(expression(everyone, CQTL['gene_1']), expression(everyone, CQTL['gene_2']),
                    s=1, color='lightgrey', label='all donors')
    strongest = info.loc[info['coexpression'].idxmax()].name
    mask = (adata.obs['ind_cov'] == strongest).values
    axes[1].scatter(expression(mask, CQTL['gene_1']), expression(mask, CQTL['gene_2']),
                    s=3, color='black', label='one individual')
    axes[1].set_xlabel(f'log({CQTL["gene_1"]} + 1)')
    axes[1].set_ylabel(f'log({CQTL["gene_2"]} + 1)')
    axes[1].legend(frameon=False, markerscale=4)


def main():
    config.set_style()
    fig, axes = plt.subplots(1, 4, figsize=(12, 2.6))
    plt.subplots_adjust(wspace=0.5)
    panel_fg(axes[:2])
    panel_hi(axes[2:])

    fig.savefig(config.figure_path('figure5FI.pdf'), bbox_inches='tight')
    fig.savefig(config.figure_path('figure5FI.png'), bbox_inches='tight', dpi=300)
    print('wrote figure5FI')


if __name__ == '__main__':
    main()
