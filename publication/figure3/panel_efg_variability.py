"""Figure 3E, 3F and 3G - baseline variability, tonic sensitivity, and the mean/variability shift.

Port of publication/original/hbec_interferon/classify_isg/select_isgs.ipynb:
  * cells 73-75   -> panel E, baseline variability of canonical vs non-canonical ISGs
  * cells 59-71   -> panel F, tonic sensitivity of canonical vs the rest
  * cells 120-122 -> panel G, change in variability against change in mean

Cells 123 and 124 left behind exact figures to check the port against: of the canonical
ISGs significantly differentially expressed at FDR < 0.01, the fraction also significant
for variability at FDR < 0.1 was 0.394 under IFN-gamma and 0.778 under IFN-beta. Both
are recomputed and reported.

Note panels E and F group genes differently, which the caption glosses over. Panel E
contrasts canonical against the non-canonical ISG modules. Panel F's second group is
every *other* gene in the macrophage tonic table -- interferon-induced genes that are
not canonical ISGs -- because that table is the universe it has values for.
"""

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns

import config
import isg_gene_lists
import memento_legacy
import reconstruct_tonic_isg

CILIATED = 'C'
TIMEPOINT = '6'
CANONICAL_LABEL = 'canonical'
OTHER_LABEL = 'non\ncanonical'
DE_FDR = 0.01
DV_FDR = 0.1


def test_path(stim):
    return config.FIGURE3_DATA + f'panelDEFG_isg/tests/{CILIATED}_{stim}_{TIMEPOINT}.h5ad'


def panel_e(ax, canonical, noncanonical):
    """Baseline (control) variability of the two ISG modules."""
    _, variability = memento_legacy.read_1d_moments_by(test_path('beta'), 'time_step')

    selected = set(canonical) | set(noncanonical)
    baseline = variability[variability['gene'].isin(selected)].copy()
    baseline['group'] = np.where(baseline['gene'].isin(canonical), CANONICAL_LABEL, OTHER_LABEL)

    sns.boxplot(x='group', y='time_step_0', data=baseline, ax=ax,
                order=[CANONICAL_LABEL, OTHER_LABEL], hue='group', legend=False,
                palette={CANONICAL_LABEL: config.CANONICAL_COLOR,
                         OTHER_LABEL: config.NONCANONICAL_COLOR})
    ax.set_xlabel(None)
    ax.set_ylabel('Baseline variability')

    groups = [baseline.query('group == @g')['time_step_0'].dropna()
              for g in [CANONICAL_LABEL, OTHER_LABEL]]
    pvalue = stats.mannwhitneyu(*groups).pvalue
    print(f'panel E  baseline variability: canonical n={len(groups[0])} '
          f'median {groups[0].median():.3f}, other n={len(groups[1])} '
          f'median {groups[1].median():.3f}, Mann-Whitney p={pvalue:.2e}')
    return pvalue


def panel_f(ax, canonical):
    """Tonic sensitivity, canonical ISGs against the rest of the tonic table."""
    tonic = reconstruct_tonic_isg.build()
    tonic['gene'] = tonic['GeneSymbol'].astype(str).str.upper()
    tonic['Tonic Sensitivity'] = np.log(tonic['Tonic Sensitivity'])
    tonic['group'] = np.where(tonic['gene'].isin(canonical), CANONICAL_LABEL, OTHER_LABEL)

    sns.boxplot(x='group', y='Tonic Sensitivity', data=tonic, ax=ax,
                order=[CANONICAL_LABEL, OTHER_LABEL], hue='group', legend=False,
                palette={CANONICAL_LABEL: config.CANONICAL_COLOR,
                         OTHER_LABEL: config.NONCANONICAL_COLOR})
    ax.set_xlabel(None)
    ax.set_ylabel('Tonic sensitivity')

    groups = [tonic.query('group == @g')['Tonic Sensitivity'].dropna()
              for g in [CANONICAL_LABEL, OTHER_LABEL]]
    pvalue = stats.mannwhitneyu(*groups).pvalue
    print(f'panel F  tonic sensitivity: canonical n={len(groups[0])} '
          f'median {groups[0].median():.3f}, other n={len(groups[1])} '
          f'median {groups[1].median():.3f}, Mann-Whitney p={pvalue:.2e}')
    return pvalue


def panel_g(axes, canonical):
    """Change in variability against change in mean, canonical ISGs highlighted."""
    fractions = {}
    for ax, stim in zip(axes, ['beta', 'gamma']):
        table = memento_legacy.read_1d_ht(test_path(stim))
        ax.scatter(table['de_coef'], table['dv_coef'], s=0.5, color='gray')

        highlighted = table.query('de_fdr < @DE_FDR & gene in @canonical')
        ax.scatter(highlighted['de_coef'], highlighted['dv_coef'], s=5,
                   color='tab:blue')
        ax.set_xlabel(r'$\Delta$ mean')
        ax.set_title(f'IFN-{stim}')
        if stim == 'beta':
            ax.set_ylabel(r'$\Delta$ variability')

        fraction = (highlighted['dv_fdr'] < DV_FDR).mean()
        fractions[stim] = fraction
        print(f'panel G  IFN-{stim}: {len(highlighted)} canonical DE genes, '
              f'{fraction:.3f} also significant for variability '
              f'(notebook reported {0.778 if stim == "beta" else 0.394:.3f})')
    return fractions


def main():
    config.set_style()
    classes = isg_gene_lists.load()
    canonical = classes.query('isg_class == "canonical"')['gene'].tolist()
    noncanonical = classes.query('isg_class == "noncanonical"')['gene'].tolist()
    print(f'{len(canonical)} canonical, {len(noncanonical)} non-canonical ISGs\n')

    fig, axes = plt.subplots(1, 2, figsize=(4, 2))
    plt.subplots_adjust(wspace=0.6)
    panel_e(axes[0], canonical, noncanonical)
    try:
        panel_f(axes[1], canonical)
    except SystemExit as reason:
        # Panel F is the one part of this figure that needs a file we do not
        # redistribute. Skipping it must not take panels E and G down with it.
        print(f'\nskipping panel F:\n{reason}\n', flush=True)
        axes[1].text(0.5, 0.5, 'Panel F\nrequires mmc2.xls\n(see README)',
                     transform=axes[1].transAxes, ha='center', va='center',
                     fontsize=7, color='firebrick', style='italic')
        axes[1].set_xticks([])
        axes[1].set_yticks([])
    fig.savefig(config.figure_path('figure3EF.pdf'), bbox_inches='tight')
    fig.savefig(config.figure_path('figure3EF.png'), bbox_inches='tight', dpi=300)

    fig, axes = plt.subplots(1, 2, figsize=(4.5, 2))
    plt.subplots_adjust(wspace=0.35)
    panel_g(axes, canonical)
    fig.savefig(config.figure_path('figure3G.pdf'), bbox_inches='tight')
    fig.savefig(config.figure_path('figure3G.png'), bbox_inches='tight', dpi=300)
    print('\nwrote figure3EF and figure3G')


if __name__ == '__main__':
    main()
