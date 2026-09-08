"""Declarative inventory of the data Figure 6 depends on.

Same contract as publication/figure{2,3,4,5}/data_manifest.py:

    python data_manifest.py check                # present on the source volume?
    python data_manifest.py check --root DIR     # ... or in a downloaded bundle
    python data_manifest.py link                 # build the tree as symlinks
    python data_manifest.py bundle --root DIR    # copy it into a standalone directory
    python data_manifest.py archive --root DIR   # ... and pack it for publication

`bundle` and `archive` require --root: they write real copies, and defaulting to the
source volume would bury its symlink tree under gigabytes of duplicates.

Figure 6 is still the lightest of the five. Panels C and D stream cells from the public
CELLxGENE census and build their own small cube; the only thing shipped is panel G's
dendritic-cell slice of the census estimators, a TileDB array (a directory, not a file)
produced by build_dc_subset.py.

This is maintainer tooling. Readers reproducing a figure download the published bundle
instead -- see publication/MAINTAINING.md.
"""

import argparse
import os
import sys

import config

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bundle_tools  # noqa: E402

REQUIRED, PROVENANCE = bundle_tools.REQUIRED, bundle_tools.PROVENANCE

CENSUS_CUBE_TAR = (config.DATA_PATH
                   + 'precomputation/stimators_cube.2023-10-23-homo_sapiens-full.tar')
DC_CUBE = 'panelG_cube/estimators_cube_dc'

REMOTE_INPUTS = [
    ('CELLxGENE census', f's3://cellxgene-data-public/cell-census/{config.CENSUS_VERSION}/',
     'streamed at run time by panels C and D, and by panel G for cell counts'),
]


def entries():
    """(panel, tier, destination in the organized tree, source path).

    One entry: the dendritic-cell cube. It is a directory, which bundle_tools copies as a
    tree. Everything else Figure 6 needs is either remote or built locally.
    """
    return [('G', REQUIRED, DC_CUBE, config.CUBE_PATH)]


def _resolve(row, root):
    return os.path.join(root, row[2]) if root else row[3]


def _tree_size(path):
    return sum(os.path.getsize(os.path.join(walk_root, name))
               for walk_root, _, names in os.walk(path) for name in names)


def check(root=None):
    rows = entries()
    ok = True
    print(f'checking {root or config.DATA_PATH} '
          f'({"organized tree" if root else "source volume"})\n')

    for panel, tier, _, _ in rows:
        path = _resolve((panel, tier, DC_CUBE, config.CUBE_PATH), root)
        if os.path.isdir(path):
            print(f'OK  panel {panel} {tier:<10} dendritic-cell cube, '
                  f'{_tree_size(path) / 1e6:.1f} MB\n    {path}')
        else:
            ok = False
            print(f'GAP panel {panel} {tier:<10} dendritic-cell cube missing\n    {path}')
            print('    Build it from the full census cube with:')
            print(f'      tar -xf {CENSUS_CUBE_TAR} \\\n'
                  f'          -C {os.path.dirname(config.CENSUS_CUBE_PATH)}')
            print('      python build_dc_subset.py')
            print(f'    census tarball present: {os.path.exists(CENSUS_CUBE_TAR)}')

    # Built locally rather than shipped: two minutes against the census, and it must use
    # the same capture rate as the full memento run it is compared against.
    comparison = config.COMPARISON_CUBE_PATH
    print(f'\npanels C, D  comparison cube (built locally)\n  {comparison}')
    if os.path.isdir(comparison):
        print(f'  OK  built, {_tree_size(comparison) / 1e6:.1f} MB')
    else:
        print('  --  not built. Build it with:')
        print('      python build_cube.py        # ~2 min')

    print('\nfetched at run time, not stored locally:')
    for name, uri, note in REMOTE_INPUTS:
        print(f'  {name}: {uri}\n      {note}')
    print(f'\nDataset {config.LUPUS_DATASET_ID} (SLE PBMC), donor {config.LUPUS_DONOR}.')
    return ok


def main():
    parser = bundle_tools.add_arguments(argparse.ArgumentParser(description=__doc__))
    bundle_tools.dispatch(parser.parse_args(), entries, check, config.FIGURE6_DATA)


if __name__ == '__main__':
    main()
