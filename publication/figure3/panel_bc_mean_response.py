"""Figure 3B and 3C - mean transcriptional response of ciliated cells to interferon.

Port of publication/original/hbec_interferon/version3/mean_var/mean_analyze.ipynb:
  * cell 16  -> panel B, LFC to IFN-alpha against LFC to beta / gamma / lambda
  * cell 21  -> the type-1 / type-2 / shared ISG classification
  * cell 35  -> panel C, per-timepoint heatmaps of LFC across the four interferons

Both read the stored hypothesis tests under hbec/binary_test_latest/ through
memento_legacy, since today's memento cannot parse results written by 0.0.6.

Timepoint note: `read_result()` in the notebook hardcodes the 3-hour files, but the
published caption says 6 hours and the cell below it sets an unused `tp = '6'`. This
script defaults to 6 to match the caption; pass --timepoint 3 to get the notebook's
literal behaviour. The classification and the panel B scatters both follow the choice.
"""

import argparse
import functools
import itertools

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns

import config
import memento_legacy

CELL_TYPE = 'C'  # ciliated
SCATTER_PAIRS = [('alpha', 'beta'), ('alpha', 'gamma'), ('alpha', 'lambda')]
LFC_LIMIT = 1000  # the notebook drops non-finite blowups before plotting


def test_path(stim, timepoint, cell_type=CELL_TYPE):
    return config.FIGURE3_DATA + f'panelBC_mean/tests/{cell_type}_{stim}_{timepoint}.h5ad'


def read_tests(timepoint):
    return {stim: memento_legacy.read_1d_ht(test_path(stim, timepoint))
            for stim in config.STIMS}


def wide_lfc(tests):
    """One row per gene, with an lfc and fdr column per interferon."""
    frames = []
    for stim, table in tests.items():
        frame = table[['gene', 'de_coef', 'de_fdr']].copy()
        frame.columns = ['gene', f'lfc_{stim}', f'fdr_{stim}']
        frames.append(frame)
    return functools.reduce(lambda a, b: a.merge(b, on='gene', how='outer'), frames)


def classify_isgs(wide):
    """Type-1 / type-2 / shared ISGs, by the LFC contrasts in notebook cell 21."""
    lfc = {stim: wide[f'lfc_{stim}'] for stim in config.STIMS}
    fdr = {stim: wide[f'fdr_{stim}'] for stim in config.STIMS}

    # Type 1: induced by alpha or beta well above what gamma and lambda manage.
    type1_best = pd.concat([lfc['alpha'], lfc['beta']], axis=1).max(axis=1)
    type1_rest = pd.concat([lfc['gamma'], lfc['lambda']], axis=1).max(axis=1)
    is_type1 = ((type1_best - type1_rest) > 1) & ((fdr['beta'] < 0.05) | (fdr['alpha'] < 0.05))

    # Type 2: gamma-specific, above both the weaker type-1 response and lambda.
    type1_worst = pd.concat([lfc['alpha'], lfc['beta']], axis=1).min(axis=1)
    type2_rest = pd.concat([type1_worst, lfc['lambda']], axis=1).max(axis=1)
    is_type2 = ((lfc['gamma'] - type2_rest) > 0.5) & (fdr['gamma'] < 0.05)

    # Shared: induced by alpha, beta and gamma alike.
    induced = pd.concat([lfc[s] for s in ['alpha', 'beta', 'gamma']], axis=1).min(axis=1) > 0.25
    significant = pd.concat([fdr[s] for s in ['alpha', 'beta', 'gamma']], axis=1).max(axis=1) < 0.05
    is_shared = induced & significant & ~is_type1 & ~is_type2

    classes = pd.DataFrame({
        'gene': wide['gene'], 'is_type1': is_type1, 'is_type2': is_type2, 'is_shared': is_shared})
    classes = classes[classes[['is_type1', 'is_type2', 'is_shared']].sum(axis=1) > 0].copy()
    classes['overall_type'] = np.select(
        [classes['is_type1'], classes['is_type2']], ['type1', 'type2'], default='shared')
    return classes


def plot_panel_b(wide, timepoint):
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.0))
    plt.subplots_adjust(wspace=0.53)

    correlations = {}
    for ax, (left, right) in zip(axes, SCATTER_PAIRS):
        subset = wide[[f'lfc_{left}', f'lfc_{right}']].dropna()
        subset = subset[(subset.max(axis=1) < LFC_LIMIT) & (subset.min(axis=1) > -LFC_LIMIT)]
        x, y = subset[f'lfc_{left}'], subset[f'lfc_{right}']
        correlations[f'{left}_vs_{right}'] = stats.pearsonr(x, y)[0]

        ax.scatter(x, y, s=1, color='grey')
        # y = x with +/- 0.5 guides, as in the notebook.
        line = np.array([x.min(), x.max()])
        ax.plot(line, line, color='k', lw=1)
        ax.plot(line, line + 0.5, '--', color='k', lw=1)
        ax.plot(line, line - 0.5, '--', color='k', lw=1)
        ax.set_xlabel(f'LFC IFN-{left}')
        ax.set_ylabel(f'LFC IFN-{right}')

    fig.savefig(config.figure_path('figure3B.pdf'), bbox_inches='tight')
    fig.savefig(config.figure_path('figure3B.png'), bbox_inches='tight', dpi=300)
    print('panel B Pearson r: ' + ', '.join(f'{k} {v:.3f}' for k, v in correlations.items()))
    return correlations


def ordered_heatmap(classes, timepoints):
    """LFC across all timepoints, with genes clustered within each ISG class."""
    per_timepoint = []
    for timepoint in timepoints:
        frames = []
        for stim in config.STIMS:
            table = memento_legacy.read_1d_ht(test_path(stim, timepoint), add_fdr=False)
            frames.append(table[['gene', 'de_coef']].rename(
                columns={'de_coef': f'lfc_{stim}_{timepoint}'}))
        merged = functools.reduce(lambda a, b: a.merge(b, on='gene'), frames)
        per_timepoint.append(merged[merged['gene'].isin(classes['gene'])])
    all_tp = functools.reduce(lambda a, b: a.merge(b, on='gene'), per_timepoint)

    ordered = []
    for isg_type in ['type1', 'type2', 'shared']:
        genes = classes.query('overall_type == @isg_type')['gene']
        block = all_tp[all_tp['gene'].isin(genes)]
        if block.empty:
            continue
        grid = sns.clustermap(block.iloc[:, 1:], row_cluster=True, col_cluster=True,
                              cmap='coolwarm', center=0, figsize=(4, 2))
        row_order = grid.dendrogram_row.reordered_ind
        plt.close('all')
        if isg_type in ('type1', 'type2'):
            row_order = row_order[::-1]
        # The notebook additionally did `row_order = row_order[300:]` for the shared
        # block. That trim is not reproduced: it was written against a larger shared
        # set, and against the ~309 shared genes here it would leave 9 rows and erase
        # the block the panel is meant to show.
        ordered.append(block.iloc[row_order, :])
        print(f'  {isg_type}: {len(genes)} classified, {block.shape[0]} in all timepoints')

    final = pd.concat(ordered, ignore_index=True)
    return final.set_index('gene')


def plot_panel_c(heatmap, timepoints):
    fig, axes = plt.subplots(1, len(timepoints), figsize=(3.6, 3.7))
    plt.subplots_adjust(wspace=0.1)
    for ax, timepoint in zip(axes, timepoints):
        block = heatmap[[f'lfc_{stim}_{timepoint}' for stim in config.STIMS]]
        sns.heatmap(block.loc[::-1], vmax=2, cmap='coolwarm', center=0,
                    xticklabels=False, yticklabels=False, cbar=False, ax=ax)
        ax.set_title(f'{timepoint}h', fontsize=8)
        ax.set_xlabel(None)
        ax.set_ylabel(None)
    fig.savefig(config.figure_path('figure3C.pdf'), bbox_inches='tight')
    fig.savefig(config.figure_path('figure3C.png'), bbox_inches='tight', dpi=300)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--timepoint', default='6', choices=config.TIMEPOINTS,
                        help='timepoint the panel B scatters and the ISG classes use')
    args = parser.parse_args()

    config.set_style()

    tests = read_tests(args.timepoint)
    wide = wide_lfc(tests)
    print(f'{wide.shape[0]} genes tested at {args.timepoint}h')

    plot_panel_b(wide, args.timepoint)

    classes = classify_isgs(wide)
    classes.to_csv(config.intermediate_path('isg_classes.csv'), index=False)
    print('ISG classes:', classes['overall_type'].value_counts().to_dict())

    heatmap = ordered_heatmap(classes, config.TIMEPOINTS)
    plot_panel_c(heatmap, config.TIMEPOINTS)
    heatmap.to_csv(config.intermediate_path('panel_c_heatmap.csv'))
    print(f'wrote figure3B and figure3C ({heatmap.shape[0]} genes)')


if __name__ == '__main__':
    main()
