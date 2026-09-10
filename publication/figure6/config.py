"""Shared paths and plotting style for the Figure 6 reproduction scripts."""

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


DATA_PATH = _dir(os.environ.get('MEMENTO_DATA_PATH', '/memento_data'))
FIGURE6_DATA = _dir(os.environ.get('FIGURE6_DATA', DATA_PATH + 'figure6_data'))

# The dendritic-cell slice of the census cube that panel G reads, ~30 MB, built by
# build_dc_subset.py and shipped in the data bundle.
CUBE_PATH = os.environ.get(
    'MEMENTO_CUBE_PATH', FIGURE6_DATA + 'panelG_cube/estimators_cube_dc')

# The full-census cube, 17 GB unpacked from precomputation/*.tar. Not in the bundle:
# panel G needs five cell types out of it, and nothing else here needs it at all. Only
# build_dc_subset.py reads this.
CENSUS_CUBE_PATH = os.environ.get(
    'MEMENTO_CENSUS_CUBE_PATH', DATA_PATH + 'precomputation/extracted/estimators_cube_v2')
# The cube panels C and D compare against, built by build_cube.py: one donor, two cell
# types, variance included, and the same capture rate as the full memento run.
COMPARISON_CUBE_PATH = os.environ.get(
    'MEMENTO_COMPARISON_CUBE_PATH',
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'intermediate',
                 'estimators_cube'))


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
