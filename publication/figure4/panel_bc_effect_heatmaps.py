"""Figure 4B and 4C - perturbation effect sizes, and how the affected genes covary.

Port of publication/perturbseq/cd4_wt_coex.ipynb cells 27-50:
  * cell 28 -> panel B, the full sgRNA-by-gene differential mean matrix
  * cell 40 -> panel C left, the same restricted to DMGs and clustered
  * cell 50 -> panel C right, WT coexpression among those DMGs

Gene and guide ordering come from agglomerative clustering at the notebook's
`distance_threshold=2`. Unlike the ISG clustering in Figure 3, this only sets display
order -- no gene membership depends on it -- so it is reproduced directly.
"""

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering

import config
import perturbseq_data

DISTANCE_THRESHOLD = 2
EFFECT_LIMIT = 0.05          # colour scale for the effect-size heatmaps
CORRELATION_LIMITS = (0, 0.5)


def short_guide(guide):
    """'IRF1.12345.ACGT' -> 'IRF1.12345', the label the notebook plotted."""
    return '.'.join(guide.split('.')[:2])


def cluster_order(matrix):
    """Rows in agglomerative-cluster order, with a colour per cluster."""
    labels = AgglomerativeClustering(
        n_clusters=None, distance_threshold=DISTANCE_THRESHOLD).fit(matrix).labels_
    order, colors, clusters = [], [], {}
    palette = sns.color_palette()
    for cluster in range(labels.max() + 1):
        members = np.where(labels == cluster)[0]
        names = matrix.index[members].tolist()
        order += names
        clusters[cluster] = names
        colors += [palette[cluster % len(palette)]] * len(members)
    return order, colors, clusters


def panel_b(effects):
    """Every tested sgRNA against every gene."""
    fig, ax = plt.subplots(figsize=(6, 3))
    sns.heatmap(effects, cmap='coolwarm', center=0,
                vmin=-EFFECT_LIMIT, vmax=EFFECT_LIMIT,
                xticklabels=False, yticklabels=False, cbar_kws={'shrink': 0.5}, ax=ax)
    ax.set_xlabel(f'{effects.shape[1]} genes')
    ax.set_ylabel(f'{effects.shape[0]} sgRNAs')
    fig.savefig(config.figure_path('figure4B.pdf'), bbox_inches='tight')
    fig.savefig(config.figure_path('figure4B.png'), bbox_inches='tight', dpi=300)


def panel_c(dmg_effects, gene_order, gene_colors, guide_order, coexpression, detailed_order):
    fig = plt.figure(figsize=(9, 4.5))

    left = fig.add_axes([0.05, 0.05, 0.32, 0.9])
    sns.heatmap(dmg_effects.T.loc[gene_order, guide_order], cmap='coolwarm', center=0,
                vmin=-EFFECT_LIMIT, vmax=EFFECT_LIMIT,
                xticklabels=False, yticklabels=False, cbar_kws={'shrink': 0.5}, ax=left)
    left.set_title('DM effect size')
    left.set_xlabel(f'{len(guide_order)} sgRNAs')
    left.set_ylabel(f'{len(gene_order)} DMGs')
    # Cluster colour strip down the left edge, standing in for clustermap row_colors.
    strip = fig.add_axes([0.02, 0.05, 0.015, 0.9])
    strip.imshow(np.array(gene_colors).reshape(-1, 1, 3), aspect='auto')
    strip.set_axis_off()

    right = fig.add_axes([0.52, 0.05, 0.42, 0.9])
    sns.heatmap(coexpression.fillna(0).loc[detailed_order, detailed_order], cmap='viridis',
                vmin=CORRELATION_LIMITS[0], vmax=CORRELATION_LIMITS[1],
                xticklabels=False, yticklabels=False, cbar_kws={'shrink': 0.5}, ax=right)
    right.set_title('WT coexpression')

    fig.savefig(config.figure_path('figure4C.pdf'), bbox_inches='tight')
    fig.savefig(config.figure_path('figure4C.png'), bbox_inches='tight', dpi=300)


def detailed_gene_order(clusters, gene_order, coexpression):
    """Reorder within each gene cluster by its coexpression structure (cell 48)."""
    order = []
    for cluster in sorted(clusters):
        members = [g for g in clusters[cluster] if g in coexpression.index]
        if len(members) > 10:
            grid = sns.clustermap(coexpression.fillna(0).loc[members, members])
            plt.close('all')
            members = [members[i] for i in grid.dendrogram_row.reordered_ind]
        order += members
    return order


def main():
    config.set_style()
    guides = perturbseq_data.selected_guides()
    effects = perturbseq_data.effect_size_matrix()
    effects = effects.loc[[g for g in effects.index if g in set(guides)]]
    print(f'{effects.shape[0]} sgRNAs x {effects.shape[1]} genes')

    panel_b(effects)

    genes = perturbseq_data.dmg_set()
    dmg_effects = effects[[g for g in genes if g in effects.columns]]
    print(f'{dmg_effects.shape[1]} DMGs present in the effect matrix')

    gene_order, gene_colors, clusters = cluster_order(dmg_effects.T)
    guide_order, _, _ = cluster_order(dmg_effects)
    print(f'{len(clusters)} gene clusters')

    coexpression = pd.read_csv(config.intermediate_path('wt_dmg_coexpression.csv'), index_col=0)
    gene_order = [g for g in gene_order if g in coexpression.index]
    gene_colors = gene_colors[:len(gene_order)]
    detailed = detailed_gene_order(clusters, gene_order, coexpression)

    panel_c(dmg_effects, gene_order, gene_colors, guide_order, coexpression, detailed)

    pd.Series(detailed, name='gene').to_csv(
        config.intermediate_path('panel_c_gene_order.csv'), index=False)
    print(f'wrote figure4B and figure4C ({len(detailed)} genes ordered)')


if __name__ == '__main__':
    main()
