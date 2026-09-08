"""Slice the full-census estimators cube down to what panel 6G reads.

Panel G compares plasmacytoid against conventional dendritic cells across 23 datasets. It
touches five cell types out of the whole census, so the 17 GB cube is almost entirely
irrelevant to it -- the slice is about 30 MB and is what the published data bundle ships.

The output keeps the source array's schema verbatim, so `panel_g_crossdataset.load_cube`
queries it exactly as it queries the full cube and nothing downstream changes.

    python build_dc_subset.py --out <dir>/estimators_cube_dc

Maintainer step. It needs the full cube, which is not part of the bundle.
"""

import argparse
import os

import numpy as np
import pandas as pd
import tiledb

import config
import panel_g_crossdataset as panel_g

WRITE_CHUNK = 250_000


def read_slice(source_uri):
    """Every dendritic-cell row for the datasets panel G lists."""
    datasets = panel_g.dataset_ids()
    frames = []
    with tiledb.open(source_uri) as cube:
        for cell_type in panel_g.DC_CELL_TYPES:
            try:
                frame = cube.df[cell_type, datasets, :]
            except tiledb.TileDBError:
                continue
            if frame.shape[0]:
                print(f'  {cell_type}: {frame.shape[0]} rows', flush=True)
                frames.append(frame)
    if not frames:
        raise SystemExit(f'no dendritic-cell rows found in {source_uri}')
    return pd.concat(frames, ignore_index=True), datasets


def write_subset(frame, source_uri, dest_uri):
    """Write `frame` into a new array carrying the source's schema unchanged."""
    with tiledb.open(source_uri) as source:
        schema = source.schema
        dims = [schema.domain.dim(i).name for i in range(schema.domain.ndim)]
        attrs = [schema.attr(i).name for i in range(schema.nattr)]

    if os.path.exists(dest_uri):
        raise SystemExit(f'{dest_uri} already exists; remove it to rebuild')
    tiledb.Array.create(dest_uri, schema)

    # Chunked so a 1.9M-row slice does not have to be handed to TileDB in one write.
    with tiledb.open(dest_uri, 'w') as dest:
        for start in range(0, frame.shape[0], WRITE_CHUNK):
            block = frame.iloc[start:start + WRITE_CHUNK]
            coords = [block[dim].to_numpy(dtype=str) for dim in dims]
            values = {attr: block[attr].to_numpy() for attr in attrs}
            dest[tuple(coords)] = values
    print(f'wrote {dest_uri}', flush=True)
    return dims, attrs


def verify(dest_uri, frame, dims, attrs):
    """Read the subset back and confirm it matches what went in."""
    with tiledb.open(dest_uri) as cube:
        readback = cube.df[:]
    if readback.shape[0] != frame.shape[0]:
        raise SystemExit(f'row count changed: wrote {frame.shape[0]}, read '
                         f'{readback.shape[0]}')

    # Sort on every column, not just the dimensions: the schema allows duplicates, and a
    # (cell type, dataset, gene, donor) key really can repeat across assays. A partial key
    # leaves tied rows in an arbitrary order and the comparison then reports a spurious
    # difference.
    key = dims + attrs
    left = frame.sort_values(key).reset_index(drop=True)
    right = readback.sort_values(key).reset_index(drop=True)
    for column in dims + attrs:
        if left[column].dtype.kind == 'f':
            if not np.allclose(left[column], right[column], equal_nan=True):
                raise SystemExit(f'column {column} differs after the round trip')
        elif not left[column].astype(str).equals(right[column].astype(str)):
            raise SystemExit(f'column {column} differs after the round trip')
    print(f'verified {readback.shape[0]} rows, {len(dims) + len(attrs)} columns')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default=config.CENSUS_CUBE_PATH,
                        help='full-census cube to slice (default: config.CENSUS_CUBE_PATH)')
    parser.add_argument('--out', default=None,
                        help='destination array (default: config.CUBE_PATH)')
    args = parser.parse_args()

    dest = args.out or config.CUBE_PATH
    if not os.path.exists(args.source):
        raise SystemExit(
            f'{args.source} not found. The full-census cube is not part of the data '
            'bundle; unpack it from precomputation/*.tar on the source volume, or pass '
            '--source.')

    os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
    frame, datasets = read_slice(args.source)
    print(f'{frame.shape[0]} rows over {frame.dataset_id.nunique()} of '
          f'{len(datasets)} datasets', flush=True)
    dims, attrs = write_subset(frame, args.source, dest)
    verify(dest, frame, dims, attrs)

    size = sum(os.path.getsize(os.path.join(walk_root, name))
               for walk_root, _, names in os.walk(dest) for name in names)
    print(f'{dest}: {size / 1e6:.1f} MB')


if __name__ == '__main__':
    main()
