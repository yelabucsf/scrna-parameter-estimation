"""Figure 5D and 5E - eQTL enrichment in cell-type-specific ATAC peaks.

Port of publication/genetics/atac_enrichment/atac_plots.ipynb cells 15-26.

Panel D asks, for eQTLs found in one cell type, how enriched they are in open chromatin
from every lineage — the diagonal should dominate if the eQTLs are genuinely cell-type
specific. Panel E takes just the matched pairs and compares the two methods head to head,
as a rank-sum z-score of eQTL p-values inside peaks against outside.

Both read enrichment results already computed on the volume. Note the stored filenames
spell the pseudobulk method `matqetl` in one directory and `mateqtl` in the other.
"""

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns

import config

HEATMAP_DIR = config.FIGURE5_DATA + 'panelDE_atac/enrichment/'
PEAKS_DIR = config.FIGURE5_DATA + 'panelDE_atac/peaks/'

# The stored heatmap files misspell the pseudobulk method.
HEATMAP_METHOD = {'memento': 'memento', 'pseudobulk': 'matqetl'}
PEAKS_METHOD = {'memento': 'memento', 'pseudobulk': 'mateqtl'}

ATAC_GROUPS = ['B', 'T', 'nk', 'myeloid']
HEATMAP_CELL_TYPES = ['B', 'T4', 'T8', 'NK', 'cM', 'ncM']
# Which ATAC lineage each cell type should match.
MATCHED_PAIRS = [('B', 'B'), ('T4', 'T'), ('T8', 'T'), ('T8', 'nk'),
                 ('NK', 'nk'), ('cM', 'myeloid'), ('ncM', 'myeloid')]
POPULATION = 'asian'


def enrichment_heatmap(population, method):
    frames = []
    for cell_type in HEATMAP_CELL_TYPES:
        path = f'{HEATMAP_DIR}{population}_{HEATMAP_METHOD[method]}_{cell_type}.out'
        frame = pd.read_csv(path, sep='\t')
        frame['ct'] = cell_type
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    combined['logp'] = -np.log10(combined['pval'])
    return (combined.pivot(index='group', columns='ct', values='logp')
            .loc[ATAC_GROUPS, HEATMAP_CELL_TYPES])


def matched_enrichment(population):
    """Rank-sum z-score for eQTL p-values inside peaks vs outside, per matched pair."""
    rows = []
    for cell_type, atac_group in MATCHED_PAIRS:
        for method in ['pseudobulk', 'memento']:
            path = f'{PEAKS_DIR}{PEAKS_METHOD[method]}/{population}_{cell_type}_{atac_group}.txt'
            data = pd.read_table(path)
            inside = data.query('in_peak == 1')['pv']
            outside = data.query('in_peak == 0')['pv']
            statistic, _ = stats.mannwhitneyu(inside, outside, alternative='less')
            n1, n2 = inside.shape[0], outside.shape[0]
            mean = n1 * n2 / 2
            sigma = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
            rows.append((cell_type, atac_group, method, (statistic - mean) / sigma))
    return pd.DataFrame(rows, columns=['ct', 'atac_group', 'method', 'zscore'])


def main():
    config.set_style()

    # --- Panel D ---
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.2))
    plt.subplots_adjust(wspace=0.35)
    for ax, method in zip(axes, ['memento', 'pseudobulk']):
        heatmap = enrichment_heatmap(POPULATION, method)
        sns.heatmap(heatmap, vmax=6, cmap='rocket', ax=ax,
                    cbar_kws={'label': '-log10(P)'})
        ax.set_title(method)
        ax.set_xlabel('eQTL cell type')
        ax.set_ylabel('ATAC lineage' if method == 'memento' else None)
        print(f'panel D {method}: diagonal mean '
              f'{np.mean([heatmap.loc[g, c] for c, g in MATCHED_PAIRS if g in heatmap.index]):.2f}')
    fig.savefig(config.figure_path('figure5D.pdf'), bbox_inches='tight')
    fig.savefig(config.figure_path('figure5D.png'), bbox_inches='tight', dpi=300)

    # --- Panel E ---
    result = matched_enrichment(POPULATION)
    fig, ax = plt.subplots(figsize=(3.2, 2.6))
    offsets = {'pseudobulk': -0.3, 'memento': 0.3}
    colors = {'memento': config.MEMENTO_COLOR, 'pseudobulk': config.PSEUDOBULK_COLOR}
    for method in ['pseudobulk', 'memento']:
        subset = result.query('method == @method')
        ax.errorbar(-subset['zscore'], np.arange(subset.shape[0]) * 2 + offsets[method],
                    xerr=1, fmt='o', capsize=3, color=colors[method], label=method)
    labels = [f'{ct} / {"mye" if group == "myeloid" else group}'
              for ct, group in MATCHED_PAIRS]
    ax.set_yticks(np.arange(len(labels)) * 2, labels)
    ax.set_xlabel('Enrichment (rank-sum z)')
    ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=2)
    for boundary in np.arange(len(labels)) * 2 + 1:
        ax.axhline(boundary, linestyle='--', lw=0.5, color='k')

    fig.savefig(config.figure_path('figure5E.pdf'), bbox_inches='tight')
    fig.savefig(config.figure_path('figure5E.png'), bbox_inches='tight', dpi=300)
    result.to_csv(config.intermediate_path('atac_matched_enrichment.csv'), index=False)

    summary = result.pivot(index=['ct', 'atac_group'], columns='method', values='zscore')
    print(summary.round(1).to_string())
    print('wrote figure5D and figure5E')


if __name__ == '__main__':
    main()
