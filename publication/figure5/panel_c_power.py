"""Figure 5C - power to recover OneK1K eQTLs, vs cohort size.

Port of publication/original/genetics/power_analysis/sample_power.ipynb.

Individuals are subsampled from the SLE cohort, and both methods are run on the same
subsample over the eQTLs a much larger cohort (OneK1K) already established. Power is the
fraction of those known eQTLs each method calls at p < 0.05. Ten resamples per cohort
size, in the four cell types the published panel shows.

Matrix eQTL tests genome-wide, so its results are joined onto memento's gene-SNP pairs
before scoring; both methods are then measured over the same set of tests.
"""

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import config

MEMENTO_DIR = config.FIGURE5_DATA + 'panelBC_replication/memento_1k/'
MATEQTL_DIR = config.FIGURE5_DATA + 'panelBC_replication/mateqtl_sampled/'

POPULATION = 'asian'
COHORT_SIZES = [50, 60, 70, 80]   # 40 is also on the volume; the panel starts at 50
NUM_RESAMPLES = 10
PANEL_CELL_TYPES = ['T4', 'B', 'cM', 'NK']
ALPHA = 0.05


def power_table():
    rows = []
    for size in COHORT_SIZES:
        for resample in range(NUM_RESAMPLES):
            for cell_type in config.CELL_TYPES:
                stem = f'{POPULATION}_{cell_type}_{size}_{resample}'
                try:
                    memento = pd.read_csv(MEMENTO_DIR + f'{stem}.csv')
                    mateqtl = pd.read_csv(MATEQTL_DIR + f'{stem}.out', sep='\t')
                except FileNotFoundError:
                    continue
                merged = memento.rename(columns={'tx': 'SNP'}).merge(
                    mateqtl, on=['SNP', 'gene'], how='left')
                rows.append((size, resample, cell_type, 'memento',
                             (memento['de_pval'] < ALPHA).mean()))
                rows.append((size, resample, cell_type, 'pseudobulk',
                             (merged['p-value'] < ALPHA).mean()))
    return pd.DataFrame(rows, columns=['num_ind', 'resample', 'ct', 'method', 'power'])


def main():
    config.set_style()
    table = power_table()
    print(f'{table.shape[0]} measurements over '
          f'{table[["num_ind", "resample", "ct"]].drop_duplicates().shape[0]} subsamples')

    fig, axes = plt.subplots(1, len(PANEL_CELL_TYPES), figsize=(8, 2.2), sharey=True)
    plt.subplots_adjust(wspace=0.25)
    palette = {'memento': config.MEMENTO_COLOR, 'pseudobulk': config.PSEUDOBULK_COLOR}

    for ax, cell_type in zip(axes, PANEL_CELL_TYPES):
        subset = table.query('ct == @cell_type')
        sns.boxplot(x='num_ind', y='power', hue='method', data=subset,
                    linewidth=1, fliersize=0, palette=palette, ax=ax)
        ax.set_title(cell_type)
        ax.set_xlabel(None)
        ax.set_ylabel(r'Power at $\alpha$ = 0.05' if cell_type == PANEL_CELL_TYPES[0] else None)
        ax.legend_.remove() if ax.get_legend() else None
    axes[len(PANEL_CELL_TYPES) // 2].set_xlabel('Number of individuals')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=2, loc='upper center',
               bbox_to_anchor=(0.5, 1.12))

    fig.savefig(config.figure_path('figure5C.pdf'), bbox_inches='tight')
    fig.savefig(config.figure_path('figure5C.png'), bbox_inches='tight', dpi=300)
    table.to_csv(config.intermediate_path('power_by_cohort_size.csv'), index=False)

    summary = table.groupby(['ct', 'num_ind', 'method'])['power'].mean().unstack()
    print(summary.round(3).to_string())
    print('wrote figure5C')


if __name__ == '__main__':
    main()
