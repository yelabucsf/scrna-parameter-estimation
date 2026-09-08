"""Shared inputs for the Figure 4 panels: which sgRNAs were tested, and their effects.

Ports the selection logic from publication/perturbseq/cd4_wt_coex.ipynb cells 11-31, which
every downstream panel depends on:

  * keep sgRNAs seen in more than 500 perturbed cells,
  * drop the wild-type (non-targeting) guides,
  * keep only regulators with usable ENCODE ChIP-seq, and
  * call differentially-expressed genes per guide, taking the ten strongest repressed
    hits from each and unioning them into the DMG set the panels display.

The notebook also read `encode_tf/files.txt` alongside the metadata; it was never used
downstream and is not in the repository, so it is skipped.
"""

import functools
import os

import pandas as pd
import scanpy as sc

import config

config.add_repo_to_path()
from memento.util import _fdrcorrect  # noqa: E402

COUNTS = config.FIGURE4_DATA + 'panelA_selection/tfko.sng.guides.full.ct.h5ad'
FILTERED_1D = config.FIGURE4_DATA + 'panelBCD_effects/filtered_1d_result.csv'
RAW_1D = config.FIGURE4_DATA + 'panelBCD_effects/raw_1d_result.csv'
ENCODE_METADATA = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'perturbseq', 'encode_tf', 'metadata.tsv')

MIN_CELLS_PER_GUIDE = 500
PEAK_TYPES = ['IDR thresholded peaks', 'optimal IDR thresholded peaks']
ASSEMBLY = 'GRCh38'
DE_FDR = 0.05
TOP_REPRESSED_PER_GUIDE = 10


def encode_regulators(candidate_tfs):
    """Regulators with a usable ENCODE peak set, by the notebook's filters."""
    meta = pd.read_csv(ENCODE_METADATA, sep='\t', header=0)
    meta = meta[meta['Output type'].isin(PEAK_TYPES) & (meta['File assembly'] == ASSEMBLY)]
    meta['target'] = meta['Experiment target'].str.split('-').str[0]
    meta = (meta.sort_values('Output type', ascending=False)
                .drop_duplicates('target')
                .query('target in @candidate_tfs'))
    meta = meta[meta['Audit ERROR'].isnull()]
    return meta['target'].tolist()


@functools.lru_cache(maxsize=1)
def selected_guides():
    """sgRNAs that pass the cell-count, non-WT and ChIP-seq availability filters."""
    obs = sc.read(COUNTS).obs
    # Guide labels carry a trailing field the tests drop.
    guides = obs['guide1_cov'].str.split('.').str[:-1].str.join('.')

    counts = guides[obs['WT'] == 'F'].value_counts()
    frequent = set(counts[counts > MIN_CELLS_PER_GUIDE].index)
    wild_type = set(guides[obs['WT'] == 'T'].drop_duplicates())
    candidates = sorted(frequent - wild_type)

    usable = set(encode_regulators([config.guide_to_gene(g) for g in candidates]))
    return [g for g in candidates if config.guide_to_gene(g) in usable]


def effect_size_matrix():
    """Guide-by-gene matrix of differential mean coefficients."""
    raw = pd.read_csv(RAW_1D)
    return raw.pivot_table('de_coef', 'tx', 'gene')


def differential_genes(guides=None):
    """Per-guide DE calls, with FDR recomputed within each guide as the notebook did."""
    guides = guides or selected_guides()
    results = pd.read_csv(FILTERED_1D).query('de_fdr < 0.1')

    frames = []
    for guide in guides:
        subset = results.query('tx == @guide').copy()
        if subset.empty:
            continue
        subset['de_fdr'] = _fdrcorrect(subset['de_pval'].values)
        frames.append(subset)
    return pd.concat(frames, ignore_index=True)


def dmg_set(per_guide=None):
    """The displayed DMGs: the strongest repressed hits from each guide, unioned."""
    per_guide = differential_genes() if per_guide is None else per_guide
    genes = set()
    for _, subset in per_guide.groupby('tx'):
        top = (subset.query('de_fdr < @DE_FDR & de_coef < 0')
                     .sort_values('de_coef')
                     .head(TOP_REPRESSED_PER_GUIDE))
        genes |= set(top['gene'])
    return sorted(genes)


if __name__ == '__main__':
    guides = selected_guides()
    print(f'{len(guides)} sgRNAs pass selection, '
          f'{len({config.guide_to_gene(g) for g in guides})} distinct regulators')
    per_guide = differential_genes(guides)
    genes = dmg_set(per_guide)
    print(f'{per_guide.shape[0]} guide-gene DE calls, {len(genes)} DMGs')
    print('DMGs:', ', '.join(genes[:15]), '...')
