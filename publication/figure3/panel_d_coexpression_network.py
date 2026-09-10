"""Figure 3D - ISG coexpression network over time.

Port of publication/original/hbec_interferon/classify_isg/select_isgs.ipynb cells 51-53.

The notebook rendered these matrices as heatmaps (`ifnb_coex_tps.png`); the published
panel shows the same data as a network, per the caption: cyan nodes are canonical ISGs,
magenta nodes non-canonical, and gene pairs with memento correlation above 0.6 are
connected. Both views are produced here -- the network as the panel, the heatmap strip
alongside it, since the heatmap is what the notebook code actually drew.

Correlations come from run_isg_clustering.py; gene classes from isg_gene_lists.py.

A single layout is computed once, from the union of all edges across timepoints, and
reused for every panel. Recomputing it per timepoint would let nodes drift and make the
change over time impossible to read.
"""

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

import config
import isg_gene_lists

CORRELATION_THRESHOLD = 0.6
PANELS = ['control'] + config.TIMEPOINTS
LAYOUT_SEED = 0


def matrix_for(panel, genes):
    name = 'gxg_control.csv' if panel == 'control' else f'gxg_beta_{panel}.csv'
    matrix = pd.read_csv(config.intermediate_path(name), index_col=0)
    keep = [g for g in matrix.index if g in genes]
    matrix = matrix.loc[keep, keep]
    np.fill_diagonal(matrix.values, 1.0)
    return matrix


def graph_from(matrix, threshold=CORRELATION_THRESHOLD):
    graph = nx.Graph()
    graph.add_nodes_from(matrix.index)
    values = matrix.values
    rows, cols = np.triu_indices(len(matrix), k=1)
    strong = values[rows, cols] > threshold
    graph.add_edges_from(zip(matrix.index[rows[strong]], matrix.index[cols[strong]]))
    return graph


def main():
    config.set_style()
    classes = isg_gene_lists.load()
    gene_class = dict(zip(classes['gene'], classes['isg_class']))
    genes = set(gene_class)

    matrices = {panel: matrix_for(panel, genes) for panel in PANELS}
    graphs = {panel: graph_from(matrix) for panel, matrix in matrices.items()}

    # One shared layout, from the union of every edge that appears at any timepoint.
    union = nx.Graph()
    union.add_nodes_from(sorted(genes))
    for graph in graphs.values():
        union.add_edges_from(graph.edges())
    layout = nx.spring_layout(union, seed=LAYOUT_SEED, k=1.2, iterations=200)

    fig, axes = plt.subplots(1, len(PANELS), figsize=(15, 2.8))
    summary = []
    for ax, panel in zip(axes, PANELS):
        graph = graphs[panel]
        colors = [config.CANONICAL_COLOR if gene_class[node] == 'canonical'
                  else config.NONCANONICAL_COLOR for node in graph.nodes()]
        nx.draw_networkx_edges(graph, layout, ax=ax, alpha=0.08, width=0.3, edge_color='grey')
        nx.draw_networkx_nodes(graph, layout, ax=ax, node_size=20, node_color=colors,
                               linewidths=0.2, edgecolors='k')
        ax.set_title('control' if panel == 'control' else f'{panel}h')
        ax.set_axis_off()
        summary.append({'panel': panel, 'genes': graph.number_of_nodes(),
                        'edges': graph.number_of_edges()})
    fig.savefig(config.figure_path('figure3D.pdf'), bbox_inches='tight')
    fig.savefig(config.figure_path('figure3D.png'), bbox_inches='tight', dpi=300)

    # The notebook's heatmap rendering of the same matrices.
    order = ([g for g in matrices['6'].index if gene_class[g] == 'canonical']
             + [g for g in matrices['6'].index if gene_class[g] == 'noncanonical'])
    fig, axes = plt.subplots(1, len(PANELS), figsize=(13, 2.4))
    for ax, panel in zip(axes, PANELS):
        matrix = matrices[panel]
        present = [g for g in order if g in matrix.index]
        sns.heatmap(matrix.loc[present, present], vmin=0.1, vmax=0.7, cmap='viridis',
                    cbar=False, xticklabels=False, yticklabels=False, ax=ax)
        ax.set_title('control' if panel == 'control' else f'{panel}h')
    fig.savefig(config.figure_path('figure3D_heatmaps.png'), bbox_inches='tight', dpi=300)

    table = pd.DataFrame(summary)
    table.to_csv(config.intermediate_path('panel_d_network_summary.csv'), index=False)
    print(table.to_string(index=False))


if __name__ == '__main__':
    main()
