"""Figure 4G - do interacting regulators share binding sites near their targets?

Port of publication/original/perturbseq/cd4_tf_coex_analysis.ipynb cells 77-84.

For every (knocked-out regulator, second regulator) pair, the differential correlation
p-values across that pair's genes are Fisher-combined into one number, and the pair is
called interacting at combined p < 0.1. Then for a range of windows around the
transcription start site, we count how many of the knockout's differentially expressed
genes have ENCODE binding sites for *both* regulators within that window.

The prediction is that interacting pairs co-bind more of their shared targets. Binding
distances come precomputed in `encode_result.csv`, so no interval arithmetic is needed.
"""

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import seaborn as sns

import config
import perturbseq_data

config.add_repo_to_path()
from memento.util import _fdrcorrect  # noqa: E402

DC_TEST_DIR = config.FIGURE4_DATA + 'panelGH_chipseq/dc_tests/'
ENCODE_RESULT = config.FIGURE4_DATA + 'panelGH_chipseq/encode_result.csv'

INTERACTION_PVALUE = 0.1
WINDOWS = [10, 50, 100, 500, 1000, 5000, 10000, 20000, 40000, 80000, 100000]
WINDOW_LABELS = ['10', '50', '100', '500', '1000', '5K', '10K', '20K', '40K', '80K', '100K']


def combined_dc_pvalues(guides):
    """Fisher-combined DC p-value per (second regulator, knockout guide)."""
    combined = {}
    for guide in guides:
        results = pd.read_csv(DC_TEST_DIR + f'{guide}_vs_WT.csv')
        results['corr_fdr'] = _fdrcorrect(results['corr_pval'].values)
        combined[guide] = results.groupby('gene_1')['corr_pval'].apply(
            lambda x: stats.combine_pvalues(x)[1])
    return pd.DataFrame(combined)


def binding_by_regulator():
    """gene -> distance-to-TSS tables, one per regulator with ENCODE data."""
    encode = pd.read_csv(ENCODE_RESULT)
    return {tf: subset[['gene', 'distance']] for tf, subset in encode.groupby('tf')}


def colocalization(guides, combined, binding, de_genes):
    rows = []
    for guide in combined.columns:
        knockout = config.guide_to_gene(guide)
        if knockout not in binding:
            continue
        targets = de_genes.get(guide, set())
        for regulator in combined.index:
            if regulator not in binding:
                continue
            pvalue = combined.loc[regulator, guide]
            if pd.isna(pvalue):
                continue
            category = ('Interacting' if pvalue < INTERACTION_PVALUE else 'Non-interacting')

            merged = binding[knockout].merge(binding[regulator], on='gene')
            merged = merged[merged['gene'].isin(targets)]
            for window in WINDOWS:
                count = ((merged['distance_x'] < window)
                         & (merged['distance_y'] < window)).sum()
                rows.append((category, count, window))
    return pd.DataFrame(rows, columns=['category', 'count', 'window'])


def main():
    config.set_style()
    guides = perturbseq_data.selected_guides()
    per_guide = perturbseq_data.differential_genes(guides)
    de_genes = {guide: set(subset['gene']) for guide, subset in per_guide.groupby('tx')}

    combined = combined_dc_pvalues(guides)
    print(f'combined DC p-values: {combined.shape[0]} regulators x {combined.shape[1]} guides')

    binding = binding_by_regulator()
    print(f'{len(binding)} regulators with ENCODE binding data')

    table = colocalization(guides, combined, binding, de_genes)
    print(table.groupby('category').size().to_dict())

    fig, ax = plt.subplots(figsize=(6, 3))
    sns.barplot(x='window', y='count', hue='category', data=table,
                errorbar=('ci', 68), capsize=0.1, palette='Set2', ax=ax)
    ax.set_xticks(range(len(WINDOWS)), WINDOW_LABELS)
    ax.set_xlabel('Window around TSS (bp)')
    ax.set_ylabel('Genes with both binding sites')
    ax.legend(frameon=False)

    fig.savefig(config.figure_path('figure4G.pdf'), bbox_inches='tight')
    fig.savefig(config.figure_path('figure4G.png'), bbox_inches='tight', dpi=300)
    table.to_csv(config.intermediate_path('chipseq_colocalization.csv'), index=False)

    means = table.groupby(['window', 'category'])['count'].mean().unstack()
    print(means.round(2).to_string())
    print('wrote figure4G')


if __name__ == '__main__':
    main()
