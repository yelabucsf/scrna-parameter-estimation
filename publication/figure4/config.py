"""Shared paths and plotting style for the Figure 4 reproduction scripts."""

import os
import sys

import matplotlib as mpl
import matplotlib.pylab as pylab

# Upstream data volume, a sync of s3://memento-paper/revision/. The original notebooks
# used '/data_volume/memento/tfko140/'.
DATA_PATH = os.environ.get('MEMENTO_DATA_PATH', '/memento_data/')
TFKO_PATH = DATA_PATH + 'tfko140/'

# Panel-organized tree built by data_manifest.py, which the panel scripts read from.
FIGURE4_DATA = os.environ.get('FIGURE4_DATA', DATA_PATH + 'figure4_data/')

# Like Figure 3, Figure 4 uses THIS repository's memento package (the notebooks import
# the memento-0.0.9 egg), not the object-oriented rewrite Figure 2 needs.
# Three levels up: publication/figureN/config.py -> the repository root. Getting this
# wrong silently falls back to a pip-installed memento in site-packages.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# github.com/mincheoly/misc-seq supplies the `encode` helper and GRCh38Genes.bed that
# the ChIP-seq panels lean on. Override with MISCSEQ_PATH.
MISCSEQ_PATH = os.environ.get('MISCSEQ_PATH', '/home/ubuntu/Github/misc-seq/miscseq')
GENE_BED = os.path.join(MISCSEQ_PATH, 'GRCh38Genes.bed')

FIGURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
INTERMEDIATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'intermediate')

# sgRNA labels are '<GENE>.<position>'; the gene is everything before the first dot.
def guide_to_gene(guide):
    return guide.split('.')[0]


def add_repo_to_path():
    for path in [REPO_ROOT, MISCSEQ_PATH]:
        if path not in sys.path:
            sys.path.insert(0, path)


def set_style():
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42
    pylab.rcParams.update({
        'legend.fontsize': 'small',
        'axes.labelsize': 'medium',
        'axes.titlesize': 'medium',
        'figure.titlesize': 'medium',
        'xtick.labelsize': 'small',
        'ytick.labelsize': 'small',
    })


def figure_path(name):
    os.makedirs(FIGURE_DIR, exist_ok=True)
    return os.path.join(FIGURE_DIR, name)


def intermediate_path(name):
    os.makedirs(INTERMEDIATE_DIR, exist_ok=True)
    return os.path.join(INTERMEDIATE_DIR, name)
