"""Shared paths and plotting style for the Figure 5 reproduction scripts."""

import os
import sys

import matplotlib as mpl
import matplotlib.pylab as pylab

# Upstream data volume, a sync of s3://memento-paper/revision/. The original notebooks
# used '/data_volume/memento/lupus/'.
DATA_PATH = os.environ.get('MEMENTO_DATA_PATH', '/memento_data/')
LUPUS_PATH = DATA_PATH + 'lupus/'

# Panel-organized tree built by data_manifest.py, which the panel scripts read from.
FIGURE5_DATA = os.environ.get('FIGURE5_DATA', DATA_PATH + 'figure5_data/')

# Three levels up: publication/figureN/config.py -> the repository root. Getting this
# wrong silently falls back to a pip-installed memento in site-packages.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MISCSEQ_PATH = os.environ.get('MISCSEQ_PATH', '/home/ubuntu/Github/misc-seq/miscseq')

FIGURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
INTERMEDIATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'intermediate')

# The two genotyped ancestry groups, and the cell types tested in each.
POPULATIONS = ['asian', 'eur']
CELL_TYPES = ['T4', 'cM', 'ncM', 'T8', 'B', 'NK']
# Minor allele frequency floor applied before any QQ plot, as in the notebooks.
MIN_ALLELE_FREQUENCY = 0.1

MEMENTO_COLOR = 'turquoise'
PSEUDOBULK_COLOR = 'slategrey'


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
        'xtick.labelsize': 'medium',
        'ytick.labelsize': 'medium',
    })


def figure_path(name):
    os.makedirs(FIGURE_DIR, exist_ok=True)
    return os.path.join(FIGURE_DIR, name)


def intermediate_path(name):
    os.makedirs(INTERMEDIATE_DIR, exist_ok=True)
    return os.path.join(INTERMEDIATE_DIR, name)
