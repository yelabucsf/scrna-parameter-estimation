"""Figure 4A - how the perturbed regulators were chosen.

Per the caption, selection rested on two criteria: expression in the target cells (top)
and binding-site availability (bottom).

No notebook in `publication/perturbseq/` draws this panel -- it documents the
experimental design, which predates the analysis code -- so it is reconstructed from the
two inputs the design used, both of which are on hand:

  * `experiment_report_2022_6_15_19h_31m.tsv`, the ENCODE query of human TF ChIP-seq
    experiments on cell lines (1,062 distinct targets) that defines the candidate pool,
  * the Perturb-seq counts, for expression of those candidates in CD4 T cells, and
  * `encode_result.csv`, for how many genes each regulator binds near.

Candidates are shown against the regulators actually perturbed, so the thresholds the
selection implies are visible rather than asserted.
"""

import os

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc

import config
import perturbseq_data

EXPERIMENT_REPORT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'perturbseq', 'experiment_report_2022_6_15_19h_31m.tsv')
ENCODE_RESULT = config.FIGURE4_DATA + 'panelGH_chipseq/encode_result.csv'
TSS_WINDOW = 10000


def candidate_regulators():
    """TFs with human ChIP-seq in ENCODE -- the pool the study selected from."""
    report = pd.read_csv(EXPERIMENT_REPORT, sep='\t', skiprows=1)
    return sorted(report['Target gene symbol'].dropna().unique())


def mean_expression(genes):
    """Mean expression per gene across all Perturb-seq cells."""
    adata = sc.read(perturbseq_data.COUNTS)
    present = [g for g in genes if g in adata.var.index]
    values = np.asarray(adata[:, present].X.mean(axis=0)).ravel()
    return pd.Series(values, index=present, name='mean_expression')


def binding_breadth(window=TSS_WINDOW):
    """Genes with a binding site within `window` of the TSS, per regulator."""
    encode = pd.read_csv(ENCODE_RESULT)
    near = encode[encode['distance'] < window]
    return near.groupby('tf')['gene'].nunique().rename('genes_bound')


def main():
    config.set_style()
    selected = sorted({config.guide_to_gene(g) for g in perturbseq_data.selected_guides()})
    candidates = candidate_regulators()
    print(f'{len(candidates)} ENCODE candidates, {len(selected)} regulators perturbed')

    expression = mean_expression(candidates)
    bound = binding_breadth()
    print(f'expression available for {expression.shape[0]} candidates; '
          f'binding breadth for {bound.shape[0]} regulators')

    fig, axes = plt.subplots(2, 1, figsize=(7, 4.5))
    plt.subplots_adjust(hspace=0.55)

    # Top: expression of every candidate, with the perturbed ones marked.
    ranked = expression.sort_values(ascending=False)
    chosen = ranked.index.isin(selected)
    axes[0].scatter(np.arange(ranked.shape[0])[~chosen], ranked.values[~chosen],
                    s=3, color='lightgrey', label='ENCODE candidate')
    axes[0].scatter(np.arange(ranked.shape[0])[chosen], ranked.values[chosen],
                    s=8, color='tab:blue', label='perturbed')
    axes[0].set_yscale('log')
    axes[0].set_xlabel('Candidate regulators, ranked by expression')
    axes[0].set_ylabel('Mean expression')
    axes[0].legend(frameon=False)
    lowest = ranked[chosen].min()
    axes[0].axhline(lowest, linestyle='--', color='k', lw=1)
    print(f'lowest-expressed perturbed regulator: {ranked[chosen].idxmin()} at {lowest:.4f}')

    # Bottom: how broadly each perturbed regulator binds.
    selected_bound = bound.reindex(selected).dropna().sort_values(ascending=False)
    axes[1].bar(np.arange(selected_bound.shape[0]), selected_bound.values,
                color='tab:blue')
    axes[1].set_xticks(np.arange(selected_bound.shape[0]))
    axes[1].set_xticklabels(selected_bound.index, rotation=90, fontsize=5)
    axes[1].set_ylabel(f'Genes bound\nwithin {TSS_WINDOW // 1000}kb of TSS')
    axes[1].set_xlabel('Perturbed regulators')

    fig.savefig(config.figure_path('figure4A.pdf'), bbox_inches='tight')
    fig.savefig(config.figure_path('figure4A.png'), bbox_inches='tight', dpi=300)

    summary = pd.concat([expression.reindex(selected), bound.reindex(selected)], axis=1)
    summary.to_csv(config.intermediate_path('regulator_selection.csv'))
    print(f'wrote figure4A ({selected_bound.shape[0]} regulators with binding data)')


if __name__ == '__main__':
    main()
