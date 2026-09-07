"""Figure 3A - UMAPs of the HTEC dataset.

Three panels, per the published caption: the whole dataset coloured by cell type, then
the ciliated cells alone coloured by stimulation and by timepoint.

Based on publication/hbec_interferon/version2/figure_4/umaps.ipynb, which predates the
final figure -- it drew only two UMAPs, both over the whole dataset (cell type and
stim). The ciliated zoom comes from the caption. Subsetting the AnnData is what
produces the zoom: the embedding is not recomputed, only the plotted cells change.
"""

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc

import config

PROCESSED = config.FIGURE3_DATA + 'panelA_umap/HBEC_type_I_processed_deep.h5ad'
CILIATED = 'ciliated'

# Shorter labels so the legends fit, following the renaming in the original notebook.
CELL_TYPE_LABELS = {'basal/club': 'basal-club', 'ionocyte/tuft': 'ion-tuft'}
# '0' is the unstimulated control; the rest are hours post-stimulation.
TIME_ORDER = ['0', '3', '6', '9', '24', '48']
# Control is drawn last so it stays visible: at 739 cells it is a twentieth of the
# ciliated population, and in stimulus order it disappears under the four interferons.
STIM_ORDER = ['alpha', 'beta', 'gamma', 'lambda', 'control']


def _ordered(series, order):
    present = [value for value in order if value in set(series)]
    return pd.Categorical(series, categories=present, ordered=True)


def main():
    config.set_style()
    adata = sc.read(PROCESSED)
    adata.obs['cell_type'] = adata.obs['cell_type'].astype(str).replace(CELL_TYPE_LABELS)
    adata.obs['stim'] = _ordered(adata.obs['stim'].astype(str), STIM_ORDER)
    adata.obs['time'] = _ordered(adata.obs['time'].astype(str), TIME_ORDER)

    ciliated = adata[adata.obs['cell_type'] == CILIATED].copy()
    print(f'{adata.shape[0]} cells total, {ciliated.shape[0]} ciliated')

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
    plt.subplots_adjust(wspace=0.45)

    sc.pl.umap(adata, color='cell_type', ax=axes[0], show=False, frameon=False,
               size=3, title='all cells', legend_fontsize=7)
    sc.pl.umap(ciliated, color='stim', ax=axes[1], show=False, frameon=False,
               size=6, title='ciliated: stimulation', legend_fontsize=7,
               palette='tab10')
    sc.pl.umap(ciliated, color='time', ax=axes[2], show=False, frameon=False,
               size=6, title='ciliated: time (h)', legend_fontsize=7,
               palette='viridis')

    fig.savefig(config.figure_path('figure3A.pdf'), bbox_inches='tight')
    fig.savefig(config.figure_path('figure3A.png'), bbox_inches='tight', dpi=300)

    summary = adata.obs['cell_type'].value_counts().rename_axis('cell_type')
    summary.rename('cells').to_csv(config.intermediate_path('panel_a_cell_counts.csv'))
    print(adata.obs['cell_type'].value_counts().to_dict())
    print('ciliated by stim:', ciliated.obs['stim'].value_counts().to_dict())
    print('wrote figure3A')


if __name__ == '__main__':
    main()
