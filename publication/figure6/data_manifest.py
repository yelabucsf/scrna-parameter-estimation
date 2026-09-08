"""Declarative inventory of the data Figure 6 depends on.

Figure 6 is unlike the others: almost nothing it needs is a file on the volume. The
comparison panels stream cells from the public CELLxGENE census at run time, and the one
large local input is the precomputed estimators cube, a TileDB array rather than a set of
files. This manifest therefore checks for the cube and reports what is fetched remotely.

    python data_manifest.py check              # is the cube unpacked and readable?
    python data_manifest.py check --root DIR   # ... against an alternative location
    python data_manifest.py link               # nothing to link; reports the same
"""

import argparse
import os
import tarfile

import config

CUBE_TAR = config.DATA_PATH + 'precomputation/stimators_cube.2023-10-23-homo_sapiens-full.tar'
CUBE_MEMBER = 'estimators_cube_v2'

REMOTE_INPUTS = [
    ('CELLxGENE census', f's3://cellxgene-data-public/cell-census/{config.CENSUS_VERSION}/',
     'streamed at run time by panels C, D and G'),
]


def cube_status(root=None):
    path = os.path.join(root, CUBE_MEMBER) if root else config.CUBE_PATH
    if not os.path.isdir(path):
        return path, False, 0
    size = sum(os.path.getsize(os.path.join(walk_root, name))
               for walk_root, _, names in os.walk(path) for name in names)
    return path, True, size


def check(root=None):
    path, present, size = cube_status(root)
    print(f'full-census estimators cube (panel G)\n  {path}')
    if present:
        print(f'  OK  unpacked, {size / 1e9:.1f} GB')
    else:
        print('  GAP not unpacked. Extract it with:')
        print(f'      mkdir -p {os.path.dirname(config.CUBE_PATH)}')
        print(f'      tar -xf {CUBE_TAR} -C {os.path.dirname(config.CUBE_PATH)}')
        print(f'  source tarball present: {os.path.exists(CUBE_TAR)}')

    # Built rather than shipped: the full-census cube carries no variance estimators, so
    # panels C and D cannot use it. See README.
    comparison = config.COMPARISON_CUBE_PATH
    print(f'\ncomparison estimators cube (panels C, D)\n  {comparison}')
    if os.path.isdir(comparison):
        size = sum(os.path.getsize(os.path.join(walk_root, name))
                   for walk_root, _, names in os.walk(comparison) for name in names)
        print(f'  OK  built, {size / 1e6:.1f} MB')
    else:
        print('  GAP not built. Build it with:')
        print('      python build_cube.py        # ~2 min')

    print('\nfetched at run time, not stored locally:')
    for name, uri, note in REMOTE_INPUTS:
        print(f'  {name}: {uri}\n      {note}')
    print(f'\nDataset {config.LUPUS_DATASET_ID} (SLE PBMC), donor {config.LUPUS_DONOR}.')
    return present


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['check', 'link'])
    parser.add_argument('--root', default=None)
    args = parser.parse_args()
    ok = check(args.root)
    if args.command == 'link':
        print('\nnothing to link: the cube is read in place and the census is remote.')
    raise SystemExit(0 if ok else 1)


if __name__ == '__main__':
    main()
