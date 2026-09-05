"""Shared paths and plotting style for the Figure 2 reproduction scripts."""

import os
import sys

import matplotlib as mpl
import matplotlib.pylab as pylab

# Root of the data volume. The original scripts used '/home/ubuntu/Data/' or
# '/data_volume/memento/'; both now live under /memento_data.
DATA_PATH = os.environ.get('MEMENTO_DATA_PATH', '/memento_data/')

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
