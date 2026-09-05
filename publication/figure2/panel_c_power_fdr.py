"""Figure 2C - power vs FDR for differential mean, variability and correlation.

Reproduces the fdr_tpr_{de,dv,dc}.pdf cells of
publication/validation/inference/simulation/{de,dv,dc}/*_plots.ipynb.
All method outputs are precomputed on the data volume under simulation/.
"""

import itertools
import pickle as pkl
from collections import OrderedDict

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc

import config

SIM_PATH = config.DATA_PATH + 'simulation/'

DM_METHODS = OrderedDict([
    ('memento', ('de/memento_wls.csv', ['coef', 'pval', 'fdr'])),
    ('edgeR LRT', ('de/edger_lrt.csv', ['logFC', 'PValue', 'FDR'])),
    ('edgeR QLF', ('de/edger_qlft.csv', ['logFC', 'PValue', 'FDR'])),
    ('t-test', ('de/t.csv', ['coef', 'pval', 'fdr'])),
])
DM_THRESHOLDS = {
    'memento': np.logspace(-5, -1, 8),
    'edgeR LRT': np.linspace(0.1, 0.7, 8),
    'edgeR QLF': np.linspace(0.1, 0.7, 8),
    't-test': np.logspace(-10, -1.5, 8),
}


def dm_curves():
    adata = sc.read(SIM_PATH + 'de/anndata.h5ad')

    results = OrderedDict()
    for name, (path, columns) in DM_METHODS.items():
        frame = pd.read_csv(SIM_PATH + path, index_col=0)[columns]
        frame.columns = ['coef', 'pval', 'fdr']
        results[name] = frame.join(adata.var[['is_de']], how='inner')

    shared = list(set.intersection(*[set(frame.index) for frame in results.values()]))

    curves = {}
    for name, frame in results.items():
        frame = frame.loc[shared]
        fdr, power = [], []
        for threshold in DM_THRESHOLDS[name]:
            hits = frame.query('pval < @threshold')
            fdr.append(1 - hits['is_de'].mean())
            power.append(hits['is_de'].sum() / frame['is_de'].sum())
        curves[name] = (fdr, power)
    return curves


def dv_curves():
    adata = sc.read(SIM_PATH + 'dv/high_expr_anndata.h5ad')
    truth = adata.var[['is_de', 'is_dv']].copy()

    memento = pd.read_csv(SIM_PATH + 'dv/memento.csv', index_col=0)
    basics = pd.read_csv(SIM_PATH + 'dv/dv_basics.csv')
    basics.index = adata.var.iloc[basics['GeneName']].index.tolist()

    result = memento.join(truth, how='left').join(basics, how='inner').dropna()
    result['is_dv'] = result['is_dv'].astype(bool)
    result['memento'] = result['pval']
    # BASiCS reports a posterior probability of differential dispersion; rank on 1 - P.
    result['basics'] = 1 - basics['ProbDiffResDisp']

    curves = {}
    for name in ['memento', 'basics']:
        fdr = np.zeros(30)
        power = np.zeros(30)
        for i, n in enumerate(np.linspace(50, 3000, 30)):
            hits = result.sort_values(name).head(int(n))['is_dv'].values
            fdr[i] = (~hits).mean()
            power[i] = hits.sum() / result['is_dv'].sum()
        curves[{'memento': 'memento', 'basics': 'BASiCS'}[name]] = (fdr, power)
    return curves


def dc_curves():
    result = pd.read_csv(SIM_PATH + 'dc/memento_dc.csv')
    schot = pd.read_csv(SIM_PATH + 'dc/dc_schot.csv')
    result = result.merge(schot, on=['gene_1', 'gene_2'])

    # The first 400 genes were simulated with a differential correlation structure.
    result['sig'] = (result['gene_1'] < 400) & (result['gene_2'] < 400)
    result['null'] = ~result['sig']
    result['schot'] = result['pvalEstimated']
    result['memento'] = result['corr_pval']
    result['schot_es'] = result['globalHigherOrderFunction']
    result['memento_es'] = result['corr_coef']

    with open(SIM_PATH + 'dc/dc_true_effect_size.pkl', 'rb') as handle:
        true_es_matrix = pkl.load(handle)
    indices = list(itertools.combinations(np.arange(true_es_matrix.shape[0]), 2))
    a, b = zip(*indices)
    truth = pd.DataFrame(indices, columns=['gene_1', 'gene_2'])
    truth['true_es'] = true_es_matrix[(a, b)]

    result = result.merge(truth, on=['gene_1', 'gene_2'], how='left')
    result['true_es'] = result['true_es'].fillna(value=0)
    for name in ['memento', 'schot']:
        result[name + '_sign'] = (result[name + '_es'] * result['true_es']) > 0

    curves = {}
    for name, label in [('memento', 'memento'), ('schot', 'scHOT')]:
        fdr, power = [], []
        for threshold in np.logspace(-3, -1, 10):
            hits = result.query(f'{name} < @threshold')
            # A hit counts only if the direction of the effect is also right.
            power.append(((result.query('sig')[name] < threshold)
                          & result.query('sig')[name + '_sign']).mean())
            fdr.append((hits['null'] | ~hits[name + '_sign']).mean())
        curves[label] = (fdr, power)
    return curves


def plot(curves, ax, title):
    colors = [config.MEMENTO_COLOR, 'slategrey', 'silver', 'lightsteelblue']
    for (label, (fdr, power)), color in zip(curves.items(), colors):
        ax.plot(fdr, power, '-o', label=label, ms=5, color=color)
    ax.set_xlabel('FDR')
    ax.set_ylabel('Power')
    ax.set_title(title)
    ax.legend(frameon=False)


def main():
    config.set_style()

    fig, axes = plt.subplots(1, 3, figsize=(8, 2.2))
    plt.subplots_adjust(wspace=0.45)
    plot(dm_curves(), axes[0], 'DM')
    plot(dv_curves(), axes[1], 'DV')
    plot(dc_curves(), axes[2], 'DC')
    fig.savefig(config.figure_path('figure2C.pdf'), bbox_inches='tight')
    fig.savefig(config.figure_path('figure2C.png'), bbox_inches='tight', dpi=300)
    print('wrote figure2C')


if __name__ == '__main__':
    main()
