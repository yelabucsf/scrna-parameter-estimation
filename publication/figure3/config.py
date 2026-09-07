"""Shared paths and plotting style for the Figure 3 reproduction scripts."""

import os
import sys

import matplotlib as mpl
import matplotlib.pylab as pylab

# Upstream data volume, a sync of s3://memento-paper/revision/. The original notebooks
# used '/data_volume/memento/hbec/' and '/data_volume/ifn_hbec/'; both are now here.
DATA_PATH = os.environ.get('MEMENTO_DATA_PATH', '/memento_data/')
HBEC_PATH = DATA_PATH + 'hbec/'

# Panel-organized tree built by data_manifest.py, which the panel scripts read from.
FIGURE3_DATA = os.environ.get('FIGURE3_DATA', DATA_PATH + 'figure3_data/')

# Figure 3 uses THIS repository's memento package, not the object-oriented rewrite that
# Figure 2 needs -- the notebooks call setup_memento / compute_1d_moments / ht_1d_moments,
# which only exist here.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FIGURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
INTERMEDIATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'intermediate')

# Per-batch capture efficiencies, from assign_q() in the original notebooks.
BATCH_CAPTURE_EFFICIENCY = {0: 0.387 * 0.25, 1: 0.392 * 0.25, 2: 0.436 * 0.25}
DEFAULT_CAPTURE_EFFICIENCY = 0.417 * 0.25

CELL_TYPE_ABBREV = {
    'basal/club': 'BC', 'basal': 'B', 'ciliated': 'C', 'goblet': 'G',
    'ionocyte/tuft': 'IT', 'neuroendocrine': 'NE',
}
STIMS = ['alpha', 'beta', 'gamma', 'lambda']
TIMEPOINTS = ['3', '6', '9', '24', '48']

CANONICAL_COLOR = 'cyan'
NONCANONICAL_COLOR = 'magenta'


def add_repo_to_path():
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)


def assign_capture_efficiency(batch):
    return BATCH_CAPTURE_EFFICIENCY.get(batch, DEFAULT_CAPTURE_EFFICIENCY)


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
