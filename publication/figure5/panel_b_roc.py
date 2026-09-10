"""Figure 5B - recovery of OneK1K eQTLs, memento against pseudobulk Matrix eQTL.

Port of publication/original/genetics/auc_curve/roc_curve.ipynb cells 39-42.

Both axes are measured rather than assumed. Power is the fraction of eQTLs already
established by the much larger OneK1K cohort that each method calls at a given p-value
threshold. The false positive rate is the fraction called at the same threshold after the
genotypes have been permuted, so a method that is simply liberal cannot look good.

Curves are averaged across the six cell types. Matrix eQTL tests genome-wide, so its
results are joined onto memento's gene-SNP pairs first; both methods are then scored over
the same tests.
"""

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc

import config

MEMENTO_DIR = config.FIGURE5_DATA + 'panelBC_replication/memento_1k/'
MATEQTL_DIR = config.FIGURE5_DATA + 'panelBC_replication/mateqtl_filtered/'
POPULATION = 'asian'
THRESHOLDS = np.linspace(0.001, 1.0, 200)


def _rates(memento, mateqtl, column_memento='de_pval', column_mateqtl='p-value'):
    """Fraction of tests called at each threshold, for both methods."""
    merged = memento.rename(columns={'tx': 'SNP'}).merge(mateqtl, on=['SNP', 'gene'])
    memento_rate = [(memento[column_memento] < t).mean() for t in THRESHOLDS]
    mateqtl_rate = [(merged[column_mateqtl] < t).mean() for t in THRESHOLDS]
    return np.array(memento_rate), np.array(mateqtl_rate)


def curves():
    memento_power, memento_fpr, mateqtl_power, mateqtl_fpr = [], [], [], []
    for cell_type in config.CELL_TYPES:
        real_memento = pd.read_csv(MEMENTO_DIR + f'{POPULATION}_{cell_type}.csv')
        real_mateqtl = pd.read_csv(MATEQTL_DIR + f'{POPULATION}_{cell_type}_filtered.out', sep='\t')
        # Permuted genotypes: everything called here is a false positive.
        null_memento = pd.read_csv(MEMENTO_DIR + f'{cell_type}_shuffled.csv')
        null_mateqtl = pd.read_csv(MATEQTL_DIR + f'{cell_type}_filtered_shuffled.out', sep='\t')

        power = _rates(real_memento, real_mateqtl)
        fpr = _rates(null_memento, null_mateqtl)
        memento_power.append(power[0])
        mateqtl_power.append(power[1])
        memento_fpr.append(fpr[0])
        mateqtl_fpr.append(fpr[1])
        print(f'  {cell_type}: {real_memento.shape[0]} OneK1K pairs, '
              f'{null_memento.shape[0]} null pairs')

    return {
        'memento': (np.vstack(memento_fpr).mean(axis=0), np.vstack(memento_power).mean(axis=0)),
        'pseudobulk': (np.vstack(mateqtl_fpr).mean(axis=0), np.vstack(mateqtl_power).mean(axis=0)),
    }


def main():
    config.set_style()
    print(f'building ROC over {len(config.CELL_TYPES)} cell types ({POPULATION})')
    roc = curves()

    fig, ax = plt.subplots(figsize=(3.2, 2.6))
    colors = {'memento': config.MEMENTO_COLOR, 'pseudobulk': config.PSEUDOBULK_COLOR}
    for method, (fpr, power) in roc.items():
        area = auc(fpr, power)
        ax.plot(fpr, power, lw=2.5, color=colors[method], label=f'{method} (AUC {area:.3f})')
        print(f'{method}: AUC {area:.4f}, power at FPR 0.05 = '
              f'{np.interp(0.05, fpr, power):.3f}')
    ax.plot([0, 1], [0, 1], '--', color='grey', lw=1)
    ax.set_xlabel('False positive rate')
    ax.set_ylabel('Power')
    ax.legend(frameon=False)

    fig.savefig(config.figure_path('figure5B.pdf'), bbox_inches='tight')
    fig.savefig(config.figure_path('figure5B.png'), bbox_inches='tight', dpi=300)

    pd.DataFrame({
        'threshold': THRESHOLDS,
        'memento_fpr': roc['memento'][0], 'memento_power': roc['memento'][1],
        'pseudobulk_fpr': roc['pseudobulk'][0], 'pseudobulk_power': roc['pseudobulk'][1],
    }).to_csv(config.intermediate_path('roc_curve.csv'), index=False)
    print('wrote figure5B')


if __name__ == '__main__':
    main()
