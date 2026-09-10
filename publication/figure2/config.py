"""Shared paths and plotting style for the Figure 2 reproduction scripts."""

import os
import sys

import matplotlib as mpl
import matplotlib.pylab as pylab


def _dir(path):
    """Normalize a directory path to end in exactly one separator.

    Every path here is built by string concatenation, so `MEMENTO_DATA_PATH=~/bundles`
    without a trailing slash would silently yield `~/bundlesfigure2_data/`. The READMEs
    tell readers to set that variable, so it has to tolerate both forms.
    """
    return os.path.join(os.path.expanduser(path), '')


# Root of the data volume. The original scripts used '/home/ubuntu/Data/' or
# '/data_volume/memento/'; both now live under /memento_data. This is the upstream
# source, organized by dataset, and is only read by data_manifest.py.
DATA_PATH = _dir(os.environ.get('MEMENTO_DATA_PATH', '/memento_data'))

# Every panel script reads from here instead: a tree organized by panel and role,
# built by `python data_manifest.py link` (symlinks) or `bundle` (a standalone 4 GB
# copy). Point this at an unpacked bundle to run without the full data volume.
FIGURE2_DATA = _dir(os.environ.get('FIGURE2_DATA', DATA_PATH + 'figure2_data'))

# The object-oriented rewrite of memento (github.com/mincheoly/memento) holds the
# estimator classes and the simulation helpers the publication scripts import.
MEMENTO_OO_PATH = os.environ.get('MEMENTO_OO_PATH', '/home/ubuntu/Github/memento')

FIGURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
INTERMEDIATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'intermediate')

MEMENTO_COLOR = 'turquoise'
BASELINE_COLOR = 'gray'


def add_memento_oo_to_path():
    if MEMENTO_OO_PATH not in sys.path:
        sys.path.insert(0, MEMENTO_OO_PATH)


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
