"""Figure 4E and 4F - the regulatory networks, and their Cytoscape inputs.

Port of publication/perturbseq/cd4_tf_coex_analysis.ipynb cells 27, 41-45, 52-64.

Panel E is the bipartite network from differential mean alone: an edge from each
regulator to every gene its knockout moves. Panel F adds differential correlation --
where knocking out one regulator changes a second regulator's correlation with a target,
the two direct edges are replaced by an interaction node joining both regulators to that
target.

The published panels were laid out in Cytoscape, so the edge lists are the reproducible
artifact; `cytoscape_SIF.csv` and `cytoscape_SIF_explicit.csv` on the volume are the
originals and this script's output is checked against them. NetworkX renderings are
emitted too, following the notebook's own layout.
"""

import itertools

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

import config
import perturbseq_data

config.add_repo_to_path()
from memento.util import _fdrcorrect  # noqa: E402

DC_RESULTS = config.FIGURE4_DATA + 'panelEF_network/guide_combine_donor.csv'
STORED_SIF = config.FIGURE4_DATA + 'panelEF_network/cytoscape_SIF.csv'
STORED_SIF_EXPLICIT = config.FIGURE4_DATA + 'panelEF_network/cytoscape_SIF_explicit.csv'

INTERACTION_FDR = 0.1
# Thresholds for the SIF edge list, from notebook cell 64.
SIF_DE_FDR = 0.001
SIF_MIN_EFFECT = 0.1
LAYOUT_SEED = 0
TF_COLOR = 'tab:blue'
INODE_COLOR = 'tab:orange'


def significant_interactions():
    """Regulator-pair interactions, FDR-corrected within each (regulator, guide) pair."""
    results = pd.read_csv(DC_RESULTS)
    results['corr_fdr'] = results.groupby(['gene_1', 'tx'])['corr_pval'].transform(
        lambda x: _fdrcorrect(x.values))
    significant = results.query('corr_fdr < @INTERACTION_FDR').copy()
    significant['knockout'] = significant['tx'].apply(config.guide_to_gene)
    return significant


def regulator_target_edges(per_guide):
    """(regulator, gene) for every gene a knockout moves -- the panel E edges."""
    edges = set()
    for guide, subset in per_guide.groupby('tx'):
        regulator = config.guide_to_gene(guide)
        edges |= {(regulator, gene) for gene in subset['gene']}
    return sorted(edges)


def build_networks(edges, interactions):
    """The DM-only graph, and the one with interaction nodes spliced in."""
    regulators = sorted({tf for tf, _ in edges})
    plain = nx.Graph()
    plain.add_nodes_from(regulators)
    plain.add_edges_from(edges)

    interacting = plain.copy()
    inodes = []
    for _, row in interactions.iterrows():
        knockout, regulator, gene = row['knockout'], row['gene_1'], row['gene_2']
        for edge in [(knockout, gene), (regulator, gene)]:
            if interacting.has_edge(*edge):
                interacting.remove_edge(*edge)
        inode = f'{knockout}+{regulator}'
        interacting.add_edges_from([(regulator, inode), (knockout, inode), (inode, gene)])
        inodes.append(inode)
    return plain, interacting, regulators, sorted(set(inodes))


def layout(graph, regulators, inodes=()):
    """Regulators along the top, targets along the bottom, interaction nodes between."""
    rng = np.random.default_rng(LAYOUT_SEED)
    targets = [n for n in graph.nodes() if n not in set(regulators) and n not in set(inodes)]
    positions = dict(zip(targets, zip(np.linspace(-1, 1, len(targets)),
                                      rng.random(len(targets)) / 100)))
    positions.update(dict(zip(regulators, zip(np.linspace(-0.9, 0.9, len(regulators)),
                                              np.ones(len(regulators))))))
    if inodes:
        positions.update(dict(zip(inodes, zip(rng.uniform(-1, 1, len(inodes)),
                                              rng.random(len(inodes)) / 10 + 0.5))))
    return positions


def draw(graph, positions, regulators, inodes, path, title):
    fig, ax = plt.subplots(figsize=(9, 6))
    regulator_set, inode_set = set(regulators), set(inodes)
    nx.draw_networkx_edges(
        graph, positions, ax=ax, width=0.15, alpha=0.25,
        edge_color=['tab:orange' if u in inode_set or v in inode_set else 'grey'
                    for u, v in graph.edges()])
    nx.draw_networkx_nodes(
        graph, positions, ax=ax,
        node_size=[18 if n in regulator_set else (5 if n in inode_set else 0.3)
                   for n in graph.nodes()],
        node_color=[TF_COLOR if n in regulator_set
                    else (INODE_COLOR if n in inode_set else 'black')
                    for n in graph.nodes()])
    labels = nx.draw_networkx_labels(
        graph, {n: (p[0], p[1] + 0.06) for n, p in positions.items()},
        labels={tf: tf for tf in regulators}, ax=ax)
    for text in labels.values():
        text.set_rotation('vertical')
        text.set_fontsize(5)
    ax.set_title(title)
    ax.set_axis_off()
    fig.savefig(path.replace('.png', '.pdf'), bbox_inches='tight')
    fig.savefig(path, bbox_inches='tight', dpi=300)
    plt.close(fig)


def write_cytoscape(interactions):
    """The SIF edge lists the published layout was built from.

    Follows notebook cell 64: regulatory edges are the strong DM calls
    (de_fdr < 0.001 and |de_coef| > 0.1), and one interaction edge per significant DC
    call. The `_explicit` variant repeats each interaction in both directions.
    """
    results = pd.read_csv(perturbseq_data.FILTERED_1D).query('de_fdr < 0.1')
    strong = results.query('de_fdr < @SIF_DE_FDR & abs(de_coef) > @SIF_MIN_EFFECT')
    regulates = [(config.guide_to_gene(row['tx']), 'regulates', row['gene'])
                 for _, row in strong.iterrows()]

    interacts, explicit = [], []
    for _, row in interactions.iterrows():
        interacts.append((row['gene_1'], 'interacts', row['knockout']))
        explicit.append((row['gene_1'], 'interacts', row['knockout']))
        explicit.append((row['knockout'], 'interacts', row['gene_1']))

    pd.DataFrame(regulates + interacts).to_csv(
        config.intermediate_path('cytoscape_SIF.csv'), index=False, header=False)
    pd.DataFrame(regulates + explicit).to_csv(
        config.intermediate_path('cytoscape_SIF_explicit.csv'), index=False, header=False)
    return len(regulates), len(interacts), len(explicit)


def compare_to_stored(regulates, interacts, explicit):
    """Check the regenerated edge lists against the ones used for the published figure."""
    for path, label, mine in [(STORED_SIF, 'cytoscape_SIF.csv', interacts),
                              (STORED_SIF_EXPLICIT, 'cytoscape_SIF_explicit.csv', explicit)]:
        stored = pd.read_csv(path, header=None, names=['source', 'relation', 'target'])
        counts = stored['relation'].value_counts().to_dict()
        match = 'MATCHES' if counts.get('interacts') == mine else 'differs'
        print(f'  {label}: stored {counts}, regenerated interacts {mine} -> {match}')
    print(f'  regenerated regulates: {regulates} (stored 14095 -- see README; the stored '
          f'file predates the thresholds in cell 64)')


def main():
    config.set_style()
    per_guide = perturbseq_data.differential_genes()
    interactions = significant_interactions()
    print(f'{per_guide.shape[0]} DM calls, {interactions.shape[0]} significant DC calls, '
          f'{interactions[["gene_1", "knockout"]].drop_duplicates().shape[0]} regulator pairs')

    edges = regulator_target_edges(per_guide)
    plain, interacting, regulators, inodes = build_networks(edges, interactions)
    print(f'panel E: {plain.number_of_nodes()} nodes, {plain.number_of_edges()} edges')
    print(f'panel F: {interacting.number_of_nodes()} nodes, '
          f'{interacting.number_of_edges()} edges, {len(inodes)} interaction nodes')

    draw(plain, layout(plain, regulators), regulators, [],
         config.figure_path('figure4E.png'), 'DM-only regulatory network')
    draw(interacting, layout(interacting, regulators, inodes), regulators, inodes,
         config.figure_path('figure4F.png'), 'with regulator interactions')

    regulates, interacts, explicit = write_cytoscape(interactions)
    compare_to_stored(regulates, interacts, explicit)
    print('wrote figure4E, figure4F and the Cytoscape edge lists')


if __name__ == '__main__':
    main()
