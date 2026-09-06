"""Figure 2B - agreement between Drop-seq estimates and smFISH ground truth.

Reproduces publication/validation/estimation/smfish/{mean,variance,correlation}/
*_comparison.ipynb. Every input is precomputed on the data volume: the smFISH
reference estimates, and the per-subsample Drop-seq estimates produced by the
sample_*_datasets.py / *_estimation.py scripts in those folders.
"""

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.stats as stats

import config

SMFISH_PATH = config.FIGURE2_DATA + 'panelB_smfish/'

PANEL_METHODS = {
    'mean': [('hypergeometric', 'memento', config.MEMENTO_COLOR, 'o', '-'),
             ('naive', 'naive', config.BASELINE_COLOR, ',', '--')],
    'variance': [('hypergeometric', 'memento', config.MEMENTO_COLOR, 'o', '-'),
                 ('poisson', 'Poisson', config.BASELINE_COLOR, 's', '-'),
                 ('basics', 'BASiCS', config.BASELINE_COLOR, '^', '-'),
                 ('naive', 'naive', config.BASELINE_COLOR, ',', '--')],
    'correlation': [('hypergeometric', 'memento', config.MEMENTO_COLOR, 'o', '-'),
                    ('poisson', 'memento (q=0)', config.BASELINE_COLOR, 's', '-'),
                    ('saver', 'SAVER', config.BASELINE_COLOR, 'v', '-'),
                    ('scvi', 'scVI', config.BASELINE_COLOR, 'd', '-'),
                    ('naive', 'naive', config.BASELINE_COLOR, ',', '--')],
}


def pearson(x, y, mask, log):
    a, b = (np.log(x[mask]), np.log(y[mask])) if log else (x[mask], y[mask])
    return mask.sum(), stats.pearsonr(a, b)[0]


def load_reference():
    ref = np.load(SMFISH_PATH + 'reference/smfish_estimates.npz', allow_pickle=True)
    dropseq_genes = sc.read_h5ad(SMFISH_PATH + 'reference/filtered_dropseq.h5ad').var.index.tolist()
    return ref, dropseq_genes


def mean_scores(ref, dropseq_genes):
    smfish_genes = list(ref['mean_genes'])
    gene_idxs = [dropseq_genes.index(g) for g in smfish_genes]
    gapdh_idx = dropseq_genes.index('GAPDH')

    means = np.load(SMFISH_PATH + 'mean/sample_means.npz')['means']
    meta = pd.read_csv(SMFISH_PATH + 'mean/sample_metadata.csv')
    # smFISH counts are reported relative to GAPDH, so match that convention.
    means = means / means[:, gapdh_idx].reshape(-1, 1)

    rows = []
    for i in range(meta.shape[0]):
        estimates = means[i][gene_idxs]
        mask = np.isfinite(np.log(ref['mean'])) & np.isfinite(np.log(estimates))
        num_valid, score = pearson(ref['mean'], estimates, mask, log=True)
        rows.append((num_valid, score))
    out = meta.copy()
    out['num_valid'], out['concordance'] = zip(*rows)
    return out


def _grouped_scores(meta, values, truth, log, min_valid, positive_mask):
    columns = [f'v{i}' for i in range(values.shape[1])]
    frame = pd.concat([meta, pd.DataFrame(values, columns=columns)], axis=1).dropna(subset=['method'])

    rows = []
    for (num_cell, trial), group in frame.groupby(['num_cell', 'trial']):
        estimates = group[columns].values
        if positive_mask:
            mask = np.all(estimates > 0, axis=0)
        else:
            mask = np.all(np.isfinite(estimates), axis=0) & np.isfinite(truth)
        if mask.sum() < 2:
            continue
        for idx, method in enumerate(group['method']):
            num_valid, score = pearson(estimates[idx], truth, mask, log=log)
            rows.append((num_cell, trial, method, num_valid, score))
    out = pd.DataFrame(rows, columns=['num_cell', 'trial', 'method', 'num_valid', 'concordance'])
    return out.query('num_valid > @min_valid')


def variance_scores(ref, dropseq_genes):
    var_genes = list(ref['var_genes'])
    gene_idxs = [dropseq_genes.index(g) for g in var_genes]
    smfish_means = np.array([m for g, m in zip(ref['mean_genes'], ref['mean']) if g in var_genes])
    smfish_cv = ref['variance'] / smfish_means ** 2

    variances = np.load(SMFISH_PATH + 'variance/sample_variances.npz')['variances']
    means = np.load(SMFISH_PATH + 'variance/sample_means.npz')['means']
    meta = pd.read_csv(SMFISH_PATH + 'variance/sample_metadata.csv')
    cv = (variances / means ** 2)[:, gene_idxs]

    return _grouped_scores(meta, cv, smfish_cv, log=True, min_valid=5, positive_mask=True)


def correlation_scores(ref):
    # Recomputed by panel_b_run_correlation.py; see that script for why the copy on
    # the data volume cannot be used.
    correlations = np.load(
        config.intermediate_path('panel_b_correlation_estimates.npz'))['correlations']
    meta = pd.read_csv(config.intermediate_path('panel_b_correlation_metadata.csv'))
    return _grouped_scores(meta, correlations, ref['correlation'], log=False, min_valid=10,
                           positive_mask=False)


def plot_curve(data, ax, color, marker, linestyle, label, positions):
    agg = data.groupby('num_cell')['concordance'].agg(['mean', 'std', 'count'])
    err = agg['std'] / np.sqrt(agg['count'])
    x = [positions[n] for n in agg.index]
    ax.plot(x, agg['mean'], marker=marker, color=color, markersize=5,
            linestyle=linestyle, label=label)
    ax.fill_between(x, agg['mean'] - err, agg['mean'] + err, alpha=0.2, color=color)


def make_panel(quantity, results, ax):
    num_cells = sorted(results['num_cell'].unique())
    positions = {n: i for i, n in enumerate(num_cells)}
    for method, label, color, marker, linestyle in PANEL_METHODS[quantity]:
        rows = results.query('method == @method')
        if rows.empty:
            continue
        plot_curve(rows, ax, color, marker, linestyle, label, positions)
    ax.set_xticks(range(len(num_cells)), [int(n) for n in num_cells], rotation=45)
    ax.set_title(quantity)
    ax.set_xlabel('number of Drop-seq cells')
    ax.set_ylabel('Pearson r vs smFISH')
    ax.legend(frameon=False, loc='upper left' if quantity == 'correlation' else 'best')


def main():
    config.set_style()
    ref, dropseq_genes = load_reference()

    tables = {
        'mean': mean_scores(ref, dropseq_genes),
        'variance': variance_scores(ref, dropseq_genes),
        'correlation': correlation_scores(ref),
    }

    fig, axes = plt.subplots(1, 3, figsize=(8, 2.2))
    plt.subplots_adjust(wspace=0.45)
    for ax, quantity in zip(axes, ['mean', 'variance', 'correlation']):
        make_panel(quantity, tables[quantity], ax)
    fig.savefig(config.figure_path('figure2B.pdf'), bbox_inches='tight')
    fig.savefig(config.figure_path('figure2B.png'), bbox_inches='tight', dpi=300)

    summary = pd.concat([t.assign(quantity=q) for q, t in tables.items()])
    summary.to_csv(config.intermediate_path('panel_b_concordance.csv'), index=False)
    print(summary.groupby(['quantity', 'num_cell', 'method'])['concordance'].mean().round(3))


if __name__ == '__main__':
    main()
