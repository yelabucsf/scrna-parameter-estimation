"""Figure 2D - concordance of single-cell DM calls with matched bulk RNA-seq.

Port of publication/validation/inference/bulk_comparison/plotting_utils.py plus the
plotting cells of bulk_comparison_plots.ipynb.

Two changes were needed against the current data volume:
  * memento's results are stored under the name 'quasiGLM', not the 'quasiML' the
    plotting helper expected, and they live alongside the other method outputs
    rather than in a local temp/ directory.
  * the Cano-Gamez bulk results are indexed by Ensembl gene ID while the single-cell
    results use symbols. The original conversion.txt is not in the repository, so the
    mapping is fetched from Ensembl BioMart and cached in intermediate/.
"""

import functools
import os
import urllib.parse
import urllib.request

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import config

CANOGAMEZ_PATH = config.DATA_PATH + 'canogamez/'
HAGAI_PATH = config.DATA_PATH + 'hagai/'
LUPUS_PATH = config.DATA_PATH + 'lupus_bulk/'

CANOGAMEZ_DATASETS = ['CD4_Memory-Th0', 'CD4_Memory-Th2', 'CD4_Memory-Th17', 'CD4_Memory-iTreg',
                      'CD4_Naive-Th0', 'CD4_Naive-Th2', 'CD4_Naive-Th17', 'CD4_Naive-iTreg']
HAGAI_DATASETS = ['Hagai2018_mouse-lps', 'Hagai2018_mouse-pic', 'Hagai2018_pig-lps',
                  'Hagai2018_rabbit-lps', 'Hagai2018_rat-lps', 'Hagai2018_rat-pic']

BULK_METHODS = [
    ('deseq2_lrt', ['log2FoldChange', 'pvalue', 'padj']),
    ('deseq2_wald', ['log2FoldChange', 'pvalue', 'padj']),
    ('edger_lrt', ['logFC', 'PValue', 'FDR']),
    ('edger_qlft', ['logFC', 'PValue', 'FDR']),
]
SC_METHODS = [
    ('quasiGLM', ['coef', 'pval', 'fdr'], 'memento'),
    ('edger_lrt', ['logFC', 'PValue', 'FDR'], 'edgeR LRT'),
    ('edger_qlft', ['logFC', 'PValue', 'FDR'], 'edgeR QLF'),
    ('deseq2_wald', ['log2FoldChange', 'pvalue', 'padj'], 'DESeq2 Wald'),
    ('deseq2_lrt', ['log2FoldChange', 'pvalue', 'padj'], 'DESeq2 LRT'),
    ('t', ['coef', 'pval', 'fdr'], 't-test'),
    ('MWU', ['coef', 'pval', 'fdr'], 'MWU'),
]
SINGLE_CELL_LABELS = {'memento', 't-test', 'MWU'}

BIOMART_QUERY = (
    '<?xml version="1.0" encoding="UTF-8"?><!DOCTYPE Query>'
    '<Query virtualSchemaName="default" formatter="TSV" header="0" uniqueRows="1" '
    'count="" datasetConfigVersion="0.6">'
    '<Dataset name="hsapiens_gene_ensembl" interface="default">'
    '<Attribute name="ensembl_gene_id"/><Attribute name="external_gene_name"/>'
    '</Dataset></Query>')


def concordance_auc(references, ranking, k=100):
    """Mean top-k overlap with each bulk reference ranking, normalised to [0, 1]."""
    total = 0
    for i in range(1, k + 1):
        total += sum(len(set(ranking[:i]) & set(ref[:i])) for ref in references) / len(references)
    return total / (k * (k + 1) / 2)


def ensembl_to_symbol():
    cache = config.intermediate_path('ensembl_gene_symbols.tsv')
    if not os.path.exists(cache):
        url = 'http://useast.ensembl.org/biomart/martservice?' + urllib.parse.urlencode(
            {'query': BIOMART_QUERY})
        with urllib.request.urlopen(url, timeout=600) as response, open(cache, 'wb') as handle:
            handle.write(response.read())
    table = pd.read_csv(cache, sep='\t', header=None, names=['ensembl', 'symbol']).dropna()
    return dict(zip(table['ensembl'], table['symbol']))


def _read(path, columns):
    frame = pd.read_csv(path, index_col=0)[columns]
    frame.columns = ['coef', 'pval', 'fdr']
    return frame


def _score_datasets(datasets, read_bulk, read_sc):
    rows = []
    for dataset in datasets:
        bulk = [read_bulk(dataset, method, columns) for method, columns in BULK_METHODS]
        single = [read_sc(dataset, method, columns) for method, columns, _ in SC_METHODS]

        shared = list(functools.reduce(
            lambda a, b: a & b, [set(frame.index) for frame in bulk + single]))
        bulk = [frame.loc[shared].sort_values('fdr') for frame in bulk]
        single = [frame.loc[shared].sort_values('fdr') for frame in single]

        references = [frame.index for frame in bulk]
        for (_, _, label), frame in zip(SC_METHODS, single):
            rows.append((label, dataset, concordance_auc(references, frame.index)))
    return pd.DataFrame(rows, columns=['name', 'dataset', 'auc'])


def hagai_scores():
    def read_bulk(dataset, method, columns):
        return _read(HAGAI_PATH + f'bulk_rnaseq/results/{dataset}_{method}.csv', columns)

    def read_sc(dataset, method, columns):
        return _read(HAGAI_PATH + f'sc_rnaseq/results/{dataset}_{method}.csv', columns)

    return _score_datasets(HAGAI_DATASETS, read_bulk, read_sc)


def canogamez_scores(trial=1):
    symbols = ensembl_to_symbol()

    def read_bulk(dataset, method, columns):
        frame = _read(CANOGAMEZ_PATH + f'bulk_results/{dataset}_{method}.csv', columns)
        frame.index = [symbols.get(gene, gene) for gene in frame.index]
        return frame

    def read_sc(dataset, method, columns):
        return _read(CANOGAMEZ_PATH + f'sc_results/{dataset}_{trial}_{method}.csv', columns)

    return _score_datasets(CANOGAMEZ_DATASETS, read_bulk, read_sc)


def lupus_scores(num_cells=100, num_trials=50):
    """The lupus dataset has no matched bulk assay, so pseudobulk-of-all-cells calls
    from four bulk-style methods stand in as the reference rankings."""
    references = [
        (f'T4_vs_cM.bulk.edger_lrt.{{}}.{{}}.csv', ['logFC', 'PValue', 'FDR']),
        (f'T4_vs_cM.bulk.edger_qlft.{{}}.{{}}.csv', ['logFC', 'PValue', 'FDR']),
        (f'T4_vs_cM.bulk.deseq2_wald.{{}}.{{}}.csv', ['log2FoldChange', 'pvalue', 'padj']),
        (f'T4_vs_cM.bulk.deseq2_lrt.{{}}.{{}}.csv', ['log2FoldChange', 'pvalue', 'padj']),
    ]
    methods = [
        ('memento', '{}_{}_quasiGLM.csv', ['coef', 'pval', 'fdr']),
        ('edgeR', 'T4_vs_cM.pseudobulk.edger_lrt.{}.{}.csv', ['logFC', 'PValue', 'FDR']),
        ('DESeq2', 'T4_vs_cM.pseudobulk.deseq2_wald.{}.{}.csv', ['log2FoldChange', 'pvalue', 'padj']),
        ('t-test', '{}_{}_t.csv', ['logFC', 'PValue', 'FDR']),
        ('MWU', '{}_{}_mwu.csv', ['logFC', 'PValue', 'FDR']),
    ]

    rows = []
    for trial in range(num_trials):
        frames = [_read(LUPUS_PATH + template.format(num_cells, trial), columns)
                  for template, columns in references]
        frames += [_read(LUPUS_PATH + template.format(num_cells, trial), columns)
                   for _, template, columns in methods]

        shared = list(functools.reduce(lambda a, b: a & b, [set(f.index) for f in frames]))
        frames = [f.loc[shared].sort_values('fdr') for f in frames]

        reference_indices = [f.index for f in frames[:len(references)]]
        for (label, _, _), frame in zip(methods, frames[len(references):]):
            rows.append((label, trial, concordance_auc(reference_indices, frame.index)))
    return pd.DataFrame(rows, columns=['name', 'trial', 'auc'])


def plot(scores, order, ax, title):
    palette = {name: (config.MEMENTO_COLOR if name in SINGLE_CELL_LABELS else 'lightcoral')
               for name in order}
    sns.boxplot(y='name', x='auc', hue='name', data=scores, order=order, palette=palette,
                legend=False, fliersize=0, ax=ax)
    sns.stripplot(y='name', x='auc', data=scores, order=order, color='grey', size=3, ax=ax)
    ax.set_title(title)
    ax.set_xlabel('concordance to bulk RNA-seq')
    ax.set_ylabel('')


def main():
    config.set_style()

    squair = pd.concat([canogamez_scores(), hagai_scores()], axis=0)
    lupus = lupus_scores()

    squair_order = ['memento', 't-test', 'MWU', 'edgeR LRT', 'edgeR QLF',
                    'DESeq2 Wald', 'DESeq2 LRT']
    lupus_order = ['memento', 't-test', 'MWU', 'edgeR', 'DESeq2']

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.6))
    plt.subplots_adjust(wspace=0.7)
    plot(squair, squair_order, axes[0], 'Cano-Gamez & Hagai')
    plot(lupus, lupus_order, axes[1], 'Perez et al. (lupus)')
    fig.savefig(config.figure_path('figure2D.pdf'), bbox_inches='tight')
    fig.savefig(config.figure_path('figure2D.png'), bbox_inches='tight', dpi=300)

    squair.to_csv(config.intermediate_path('panel_d_squair_auc.csv'), index=False)
    lupus.to_csv(config.intermediate_path('panel_d_lupus_auc.csv'), index=False)
    print(squair.groupby('name')['auc'].mean().round(3))
    print(lupus.groupby('name')['auc'].mean().round(3))


if __name__ == '__main__':
    main()
