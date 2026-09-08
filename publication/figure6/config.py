"""Shared paths and plotting style for the Figure 6 reproduction scripts."""

import os
import sys

import matplotlib as mpl
import matplotlib.pylab as pylab

DATA_PATH = os.environ.get('MEMENTO_DATA_PATH', '/memento_data/')
# The precomputed estimators cube, unpacked from precomputation/*.tar.
CUBE_PATH = os.environ.get(
    'MEMENTO_CUBE_PATH', DATA_PATH + 'precomputation/extracted/estimators_cube_v2')

FIGURE6_DATA = os.environ.get('FIGURE6_DATA', DATA_PATH + 'figure6_data/')

# Three levels up: publication/figureN/config.py -> the repository root.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FIGURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
INTERMEDIATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'intermediate')

# The SLE PBMC dataset the comparison panels use, and the donor within it.
LUPUS_DATASET_ID = '218acb0f-9f2f-4f76-b90b-15a4b7c7f629'
LUPUS_DONOR = '1259'
# The notebook used census 2023-10-30, which CZI has since retired. 2023-12-15 is the
# nearest surviving release; the Lupus dataset is byte-identical in cell count across
# every available release, so the comparison is unaffected.
CENSUS_VERSION = os.environ.get('CENSUS_VERSION', '2023-12-15')

COMPARISON_CELL_TYPES = ('classical monocyte', 'CD4-positive, alpha-beta T cell')
CAPTURE_RATE = 0.07

MEMENTO_COLOR = 'turquoise'
BASELINE_COLOR = 'slategrey'


def add_repo_to_path():
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)


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
