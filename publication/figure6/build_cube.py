"""Build the precomputed estimators cube that panels 6C and 6D compare against.

Panels C and D ask whether memento's precomputed mode agrees with a full run. That needs
a cube carrying the *variance* estimators, and the cube distributed on the volume does
not have them (see README). This rebuilds one from the public CELLxGENE census, scoped to
the single donor and two cell types the comparison uses -- about two minutes and 1.2 MB,
against the 17 GB of the full census build.

The estimators themselves come from memento-cxg, used unmodified; this module only
supplies the scope, the capture rate, and the anonymous-S3 context that its `run()` entry
point does not set at the top level.

    python build_cube.py            # writes intermediate/estimators_cube

Requires a memento-cxg checkout; point at it with MEMENTO_CXG_PATH.
"""

import argparse
import logging
import multiprocessing
import os
import sys

import config

MEMENTO_CXG_PATH = os.environ.get('MEMENTO_CXG_PATH',
                                  os.path.expanduser('~/Github/memento-cxg'))
CENSUS_URI = ('s3://cellxgene-data-public/cell-census/'
              f'{config.CENSUS_VERSION}/soma/census_data/homo_sapiens')
CUBE_NAME = 'estimators_cube'
SIZE_FACTOR_NAME = 'obs_with_size_factor'


def import_builder(capture_rate):
    """Import memento-cxg's builder, with the capture rate already in the environment.

    Pass 2 runs in *spawned* processes, which re-import the builder module and so never
    see an attribute assigned on it here -- only the environment survives the process
    boundary. So the capture rate has to be set before the import, and the builder has to
    read it from there.
    """
    os.environ['MEMENTO_CUBE_Q'] = str(capture_rate)
    if not os.path.isdir(MEMENTO_CXG_PATH):
        raise SystemExit(
            f'memento-cxg not found at {MEMENTO_CXG_PATH}.\n'
            'Clone https://github.com/mincheoly/memento-cxg and set MEMENTO_CXG_PATH.')
    sys.path.insert(0, MEMENTO_CXG_PATH)
    from memento import cell_census_summary_cube as builder

    if builder.Q != capture_rate:
        raise SystemExit(
            f'memento-cxg ignored MEMENTO_CUBE_Q: its Q is {builder.Q}, expected '
            f'{capture_rate}. This checkout predates the change making the capture rate '
            'configurable; a cube built with it would silently use the wrong q. See the '
            'README section "memento-cxg patches".')
    check_numpy2(builder)
    return builder


def check_numpy2(builder):
    """Fail now, not once per gene in every worker, if the numpy 2 fix is missing."""
    import numpy as np
    import scipy.sparse
    from memento.estimators import compute_variance

    counts = scipy.sparse.csc_matrix(np.array([[0], [1], [2], [5]], dtype=float))
    try:
        compute_variance(counts, builder.Q, np.array([1.0, 1.0, 1.0, 1.0]))
    except TypeError as error:
        raise SystemExit(
            f'memento-cxg is not numpy-2 compatible: {error}\n'
            'compute_variance returns a size-1 array where a scalar is required. See the '
            'README section "memento-cxg patches".')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default=config.intermediate_path(''),
                        help='directory to write the cube into (default: intermediate/)')
    parser.add_argument('--capture-rate', type=float, default=config.CAPTURE_RATE,
                        help='q, the assumed capture efficiency. Must match the value the '
                             'full memento run uses, or the two routes are not comparable '
                             '(default: config.CAPTURE_RATE)')
    args = parser.parse_args()

    if multiprocessing.get_start_method(True) != 'spawn':
        multiprocessing.set_start_method('spawn', True)

    builder = import_builder(args.capture_rate)

    import tiledb
    import tiledbsoma as soma
    from somacore import AxisQuery

    types = ' or '.join(f"cell_type == '{c}'" for c in config.COMPARISON_CELL_TYPES)
    builder.OBS_VALUE_FILTER = (
        f"is_primary_data == True and dataset_id == '{config.LUPUS_DATASET_ID}' "
        f"and donor_id == '{config.LUPUS_DONOR}' and ({types})")
    os.makedirs(args.out, exist_ok=True)
    builder.ESTIMATORS_CUBE_ARRAY_URI = os.path.join(args.out, CUBE_NAME)
    builder.OBS_WITH_SIZE_FACTOR_TILEDB_ARRAY_URI = os.path.join(args.out, SIZE_FACTOR_NAME)
    builder.MAX_WORKERS = 8

    if tiledb.array_exists(builder.ESTIMATORS_CUBE_ARRAY_URI):
        raise SystemExit(f'{builder.ESTIMATORS_CUBE_ARRAY_URI} already exists. '
                         'Remove it to rebuild -- the builder appends, so reusing the '
                         'path would duplicate every row.')

    logging.info(f'q={builder.Q}  census={config.CENSUS_VERSION}')
    logging.info(f'cube -> {builder.ESTIMATORS_CUBE_ARRAY_URI}')

    context = soma.SOMATileDBContext().replace(tiledb_config={
        'soma.init_buffer_bytes': builder.TILEDB_SOMA_BUFFER_BYTES,
        'vfs.s3.region': 'us-west-2',
        'vfs.s3.no_sign_request': True})

    with soma.Experiment.open(uri=CENSUS_URI, context=context) as experiment:
        query = experiment.axis_query(
            measurement_name='RNA',
            obs_query=AxisQuery(value_filter=builder.OBS_VALUE_FILTER),
            # All genes, so the size factors are computed over the whole transcriptome.
            var_query=AxisQuery())
        logging.info(f'Pass 1: {query.n_obs} cells and {query.n_vars} genes')

        if tiledb.array_exists(builder.OBS_WITH_SIZE_FACTOR_TILEDB_ARRAY_URI):
            size_factors = tiledb.open(
                builder.OBS_WITH_SIZE_FACTOR_TILEDB_ARRAY_URI).df[:].set_index('soma_joinid')
        else:
            size_factors = builder.pass_1_compute_size_factors(query, 'raw')
            tiledb.from_pandas(builder.OBS_WITH_SIZE_FACTOR_TILEDB_ARRAY_URI,
                               size_factors.reset_index(), index_col=[0])

        query = experiment.axis_query(
            measurement_name='RNA',
            obs_query=AxisQuery(value_filter=builder.OBS_VALUE_FILTER),
            var_query=AxisQuery(value_filter=builder.VAR_VALUE_FILTER))
        builder.pass_2_compute_estimators(query, size_factors,
                                          measurement_name='RNA', layer='raw')

    with tiledb.open(builder.ESTIMATORS_CUBE_ARRAY_URI) as cube:
        frame = cube.df[:]
    with_variance = int((frame['var'] > 0).sum())
    print(f'wrote {builder.ESTIMATORS_CUBE_ARRAY_URI}: {len(frame)} rows, '
          f'{with_variance} ({100 * with_variance / len(frame):.1f}%) with a variance')
    # Genes too sparse to estimate a second moment are left at zero by design, so a
    # healthy build still leaves most rows at zero -- but nowhere near all of them.
    if with_variance / len(frame) < panel_threshold():
        print('WARNING: variance coverage is low enough that panel D will not be a real '
              'comparison. Check that memento-cxg computes variance and sev.')


def panel_threshold():
    import panel_cd_comparison
    return panel_cd_comparison.MIN_VARIANCE_COVERAGE


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                        level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
    main()
