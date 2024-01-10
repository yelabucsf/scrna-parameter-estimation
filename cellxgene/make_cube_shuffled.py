import concurrent
import gc
import logging
import multiprocessing
import os
import sys
from concurrent import futures
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import scipy.sparse
import scipy.sparse
import tiledb
import tiledbsoma as soma
from somacore import ExperimentAxisQuery, AxisQuery
from tiledb import ZstdFilter, ArraySchema, Domain, Dim, Attr, FilterList

from .estimators import compute_mean, compute_sem, bin_size_factor, compute_sev, compute_variance, gen_multinomial

TEST_MODE = bool(os.getenv("TEST_MODE", False))  # Read data from simple test fixture Census data
PROFILE_MODE = bool(os.getenv("PROFILE_MODE", False))  # Run pass 2 in single-process mode with profiling output

ESTIMATORS_CUBE_ARRAY_URI = "estimators_cube_shuffled"

OBS_WITH_SIZE_FACTOR_TILEDB_ARRAY_URI = "obs_with_size_factor_2"

TILEDB_SOMA_BUFFER_BYTES = 2**31
if TEST_MODE:
    TILEDB_SOMA_BUFFER_BYTES = 10 * 1024 ** 2

# The minimum number of cells that should be processed at a time by each child process.
MIN_BATCH_SIZE = 2**14
# For testing
MIN_BATCH_SIZE = 5000

CUBE_TILEDB_DIMS_OBS = [
    "cell_type",
    "dataset_id",
]

CUBE_TILEDB_ATTRS_OBS = [
    "assay",
    "suspension_type",
    "donor_id",
    "disease",
    "sex"
]

if TEST_MODE:
    CUBE_TILEDB_DIMS_OBS = ["celltype"]
    CUBE_TILEDB_ATTRS_OBS = ["study"]

CUBE_LOGICAL_DIMS_OBS = CUBE_TILEDB_DIMS_OBS + CUBE_TILEDB_ATTRS_OBS

CUBE_DIMS_VAR = ['feature_id']

if TEST_MODE:
    CUBE_DIMS_VAR = ['var_id']

CUBE_TILEDB_DIMS = CUBE_DIMS_VAR + CUBE_TILEDB_DIMS_OBS

ESTIMATOR_NAMES = ['nnz', 'n_obs', 'min', 'max', 'sum', 'mean', 'sem', 'var', 'sev', 'selv']


CUBE_SCHEMA = ArraySchema(
  domain=Domain(*[
    Dim(name=dim_name, dtype="ascii", filters=FilterList([ZstdFilter(level=-1), ]))
    for dim_name in CUBE_TILEDB_DIMS
  ]),
  attrs=[Attr(name=attr_name, dtype='ascii', nullable=False, filters=FilterList([ZstdFilter(level=-1), ]))
         for attr_name in CUBE_TILEDB_ATTRS_OBS] +
        [Attr(name=estimator_name, dtype='float64', var=False, nullable=False, filters=FilterList([ZstdFilter(level=-1), ]))
         for estimator_name in ESTIMATOR_NAMES],
  cell_order='row-major',
  tile_order='row-major',
  capacity=10000,
  sparse=True,
  allows_duplicates=True,
)

Q = 0.07  # RNA capture efficiency depending on technology

MAX_WORKERS = 30  # None means use multiprocessing's dynamic default

VAR_VALUE_FILTER = None
# For testing. Note this only affects pass 2, since all genes must be considered when computing size factors in pass 1.
# VAR_VALUE_FILTER = "feature_id == 'ENSG00000000419'" #ENSG00000002330'"

# OBS_VALUE_FILTER = "is_primary_data == True"
# For testing
# OBS_VALUE_FILTER = "is_primary_data == True and tissue_general == 'embryo'"

dataset_ids_to_query = ['c7775e88-49bf-4ba2-a03b-93f00447c958', '218acb0f-9f2f-4f76-b90b-15a4b7c7f629', '4c4cd77c-8fee-4836-9145-16562a8782fe']
dataset_query = '('
for idx, di in enumerate(dataset_ids_to_query): 
    dataset_query += f'dataset_id == "{di}" '
    if idx != len(dataset_ids_to_query)-1:
        dataset_query += 'or '
dataset_query += ')'

celltypes_to_query = [
    'conventional dendritic cell',
    'plasmacytoid dendritic cell',
    'conventional dendritic cell',
    'plasmacytoid dendritic cell, human',
    'dendritic cell',
    'dendritic cell, human',
    'myeloid dendritic cell',
    'plasmacytoid dendritic cell']
celltype_query = '('
for idx, ct in enumerate(celltypes_to_query): 
    celltype_query += f'cell_type == "{ct}" '
    if idx != len(celltypes_to_query)-1:
        celltype_query += 'or '
celltype_query += ')'

OBS_VALUE_FILTER_1 = dataset_query # All cells in three datasets
OBS_VALUE_FILTER_2 = dataset_query + ' and ' + celltype_query # only relevant celltypes

# OBS_VALUE_FILTER = "is_primary_data == True and (dataset_id ==  '218acb0f-9f2f-4f76-b90b-15a4b7c7f629' and donor_id == '1259')" # Lupus dataset
# OBS_VALUE_FILTER = "is_primary_data == True and cell_type == 'CD14-positive monocyte' and dataset_id ==  '86282760-5099-4c71-8cdd-412dcbbbd0b9'"
# OBS_VALUE_FILTER = "is_primary_data == True and (cell_type == 'CD14-positive monocyte' or cell_type == 'dendritic cell') and (dataset_id == '1a2e3350-28a8-4f49-b33c-5b67ceb001f6' or dataset_id == '3faad104-2ab8-4434-816d-474d8d2641db')"

if TEST_MODE:
    OBS_VALUE_FILTER = None


logging.basicConfig(
    format="%(asctime)s %(process)-7s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logging.captureWarnings(True)


pd.options.display.max_columns = None
pd.options.display.width = 1024
pd.options.display.min_rows = 40


def compute_all_estimators_for_obs_group(obs_group, obs_df):
    """Computes all estimators for a given {cell type, dataset} group of expression values"""
    size_factors_for_obs_group = obs_df[
        (obs_df[CUBE_LOGICAL_DIMS_OBS[0]] == obs_group.name[0]) &
        (obs_df[CUBE_LOGICAL_DIMS_OBS[1]] == obs_group.name[1])][['approx_size_factor']]
    gene_groups = obs_group.groupby(CUBE_DIMS_VAR)
    estimators = gene_groups.apply(lambda gene_group: compute_all_estimators_for_gene(obs_group.name, gene_group, size_factors_for_obs_group))
    return estimators


def compute_all_estimators_for_gene(obs_group_name: str, gene_group: pd.DataFrame, size_factors_for_obs_group: pd.DataFrame):
    """Computes all estimators for a given {cell type, dataset, gene} group of expression values"""
    group_name = (*obs_group_name, gene_group.name)

    data_dense = (
        size_factors_for_obs_group[[]].  # just the soma_dim_0 index
        join(gene_group[['soma_dim_0', 'soma_data']].set_index('soma_dim_0'), how='left').
        reset_index()
    )

    X_dense = data_dense.soma_data.to_numpy()
    X_dense = np.nan_to_num(X_dense)
    size_factors_dense = size_factors_for_obs_group.approx_size_factor.to_numpy()

    data_sparse = data_dense[data_dense.soma_data.notna()]
    X_sparse = data_sparse.soma_data.to_numpy()
    X_csc = scipy.sparse.coo_array((X_sparse, (data_sparse.index, np.zeros(len(data_sparse), dtype=int))),
                                   shape=(len(data_dense), 1)).tocsc()

    n_obs = len(X_dense)
    if n_obs == 0:
        return pd.Series(data=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    nnz = gene_group.shape[0]
    min_ = X_sparse.min()
    max_ = X_sparse.max()
    sum_ = X_sparse.sum()
    mean = compute_mean(X_dense, size_factors_dense)
    sem = compute_sem(X_dense, size_factors_dense)
    variance = compute_variance(X_csc, Q, size_factors_dense, group_name=group_name)
    sev, selv = compute_sev(X_csc, Q, size_factors_dense, num_boot=500, group_name=group_name)

    return pd.Series(data=[nnz, n_obs, min_, max_, sum_, mean, sem, variance, sev, selv])


def compute_all_estimators_for_batch_tdb(soma_dim_0, obs_df: pd.DataFrame, var_df: pd.DataFrame, X_uri: str,
                                         batch: int) -> pd.DataFrame:
    """Compute estimators for each gene"""

    with soma.SparseNDArray.open(
        X_uri, 
        context=soma.SOMATileDBContext().replace(tiledb_config={
            "soma.init_buffer_bytes": TILEDB_SOMA_BUFFER_BYTES,
            "vfs.s3.region":"us-west-2",
            "vfs.s3.no_sign_request":True})) as X:
        X_df = X.read(coords=(soma_dim_0, var_df.index.values)).tables().concat().to_pandas()
        logging.info(f"Pass 2: Start X batch {batch}, cells={len(soma_dim_0)}, nnz={len(X_df)}")
        result = compute_all_estimators_for_batch_pd(X_df, obs_df, var_df)
        if len(result) == 0:
            logging.warning(f"Pass 2: Batch {batch} had empty result, cells={len(soma_dim_0)}, nnz={len(X_df)}")
        logging.info(f"Pass 2: End X batch {batch}, cells={len(soma_dim_0)}, nnz={len(X_df)}")

    gc.collect()

    return result


def compute_all_estimators_for_batch_pd(X_df: pd.DataFrame, obs_df: pd.DataFrame, var_df: pd.DataFrame):
    result = (
        X_df.merge(var_df[CUBE_DIMS_VAR], left_on='soma_dim_1', right_index=True).
        merge(obs_df[CUBE_LOGICAL_DIMS_OBS], left_on='soma_dim_0', right_index=True).
        drop(columns=['soma_dim_1']).
        groupby(CUBE_LOGICAL_DIMS_OBS, observed=True, sort=False).
        apply(
            lambda obs_group: compute_all_estimators_for_obs_group(obs_group, obs_df)).
        rename(mapper=dict(enumerate(ESTIMATOR_NAMES)), axis=1)
    )
    return result


def sum_gene_expression_levels_by_cell(X_tbl: pa.Table, batch: int) -> pd.Series:
    logging.info(f"Pass 1: Computing X batch {batch}, nnz={X_tbl.shape[0]}")

    # TODO: use PyArrow API only; avoid Pandas conversion
    result = (
        X_tbl.
        to_pandas()[['soma_dim_0', 'soma_data']].
        groupby('soma_dim_0', sort=False).
        sum()['soma_data']
    )

    logging.info(f"Pass 1: Computing X batch {batch}, nnz={X_tbl.shape[0]}: done")

    return result


def pass_1_compute_size_factors(query: ExperimentAxisQuery, layer: str) -> pd.DataFrame:
    obs_df = (
        query.obs(column_names=["soma_joinid"] + CUBE_LOGICAL_DIMS_OBS).
        concat().
        to_pandas().
        set_index("soma_joinid")
    )
    obs_df['size_factor'] = 0  # accumulated

    executor = futures.ThreadPoolExecutor()
    summing_futures = []
    X_rows = query._ms.X[layer].shape[0]
    cum_rows = 0
    for n, X_tbl in enumerate(query.X(layer).tables(), start=1):
        cum_rows += X_tbl.shape[0]
        logging.info(f"Pass 1: Submitting X batch {n}, nnz={X_tbl.shape[0]}, {100 * cum_rows / X_rows:0.1f}%")
        summing_futures.append(executor.submit(sum_gene_expression_levels_by_cell, X_tbl, n))

    for n, summing_future in enumerate(futures.as_completed(summing_futures), start=1):
        # Accumulate cell sums, since a given cell's X values may be returned across multiple tables
        cell_sums = summing_future.result()
        obs_df['size_factor'] = obs_df['size_factor'].add(cell_sums, fill_value=0)
        logging.info(f"Pass 1: Completed {n} of {len(summing_futures)} batches, "
                     f"total cube rows={len(obs_df)}")
    
    # Convert size factors to relative - prevents small floats for variance
    global_n_umi = obs_df['size_factor'].values.mean()
    obs_df['size_factor'] = obs_df['size_factor'].values#/global_n_umi

    # Bin all sums to have fewer unique values, to speed up bootstrap computation
    obs_df['approx_size_factor'] = bin_size_factor(obs_df['size_factor'].values)

    return obs_df[CUBE_LOGICAL_DIMS_OBS + ['approx_size_factor']]


def pass_2_compute_estimators(query: ExperimentAxisQuery, size_factors: pd.DataFrame, /,
                              measurement_name: str, layer: str) -> None:
    var_df = query.var().concat().to_pandas().set_index("soma_joinid")
    obs_df = query.obs(column_names=['soma_joinid'] + CUBE_LOGICAL_DIMS_OBS).concat().to_pandas().set_index("soma_joinid")
    obs_df = obs_df.join(size_factors[['approx_size_factor']])
    
    # SHUFFLE CELL TYPE LABELS HERE
    obs_df['cell_type'] = obs_df['cell_type'].sample(frac=1).values

    # accumulate into a TileDB array
    tiledb.Array.create(ESTIMATORS_CUBE_ARRAY_URI, CUBE_SCHEMA, overwrite=True)

    # Process X by cube rows. This ensures that estimators are computed
    # for all X data contributing to a given cube row aggregation.
    # TODO: `groups` converts categoricals to strs, which is inefficient
    cube_obs_coords = obs_df[CUBE_LOGICAL_DIMS_OBS].groupby(CUBE_LOGICAL_DIMS_OBS)
    cube_obs_coord_groups = cube_obs_coords.groups

    soma_dim_0_batch = []
    batch_futures = []
    n = n_cum_cells = 0
    executor = futures.ProcessPoolExecutor(max_workers=MAX_WORKERS)
    n_total_cells = query.n_obs

    # For testing/debugging: Run pass 2 without multiprocessing
    if PROFILE_MODE:
        # force numba jit compilation outside of profiling
        gen_multinomial(np.array([1, 1, 1]), 3, 1)

        import cProfile

        def process_batch():
            nonlocal n
            n += 1
            batch_result = compute_all_estimators_for_batch_tdb(soma_dim_0_batch, obs_df, var_df,
                                                                query.experiment.ms[measurement_name].X[layer].uri, n)
            if len(batch_result) > 0:
                tiledb.from_pandas(ESTIMATORS_CUBE_ARRAY_URI, batch_result.reset_index(CUBE_TILEDB_ATTRS_OBS), mode='append')

        with cProfile.Profile() as pr:
            for soma_dim_0_ids in cube_obs_coord_groups.values():
                soma_dim_0_batch.extend(soma_dim_0_ids)
                if len(soma_dim_0_batch) < MIN_BATCH_SIZE:
                    continue

                process_batch()
                soma_dim_0_batch = []

            if len(soma_dim_0_batch) > 0:
                process_batch()

            pr.dump_stats(f"pass_2_compute_estimators_{n}.prof")

    else:  # use multiprocessing
        def submit_batch(soma_dim_0_batch_):
            nonlocal n, n_cum_cells
            n += 1
            n_cum_cells += len(soma_dim_0_batch_)
            logging.info(f"Pass 2: Submitting cells batch {n}, cells={len(soma_dim_0_batch)}, "
                         f"{100 * n_cum_cells / n_total_cells:0.1f}%")
            batch_futures.append(executor.submit(compute_all_estimators_for_batch_tdb,
                                                 soma_dim_0_batch_,
                                                 obs_df,
                                                 var_df,
                                                 query.experiment.ms[measurement_name].X[layer].uri,
                                                 n))

        for soma_dim_0_ids in cube_obs_coord_groups.values():
            soma_dim_0_batch.extend(soma_dim_0_ids)

            # Fetch data for multiple cube rows at once, to reduce X.read() call count
            if len(soma_dim_0_batch) < MIN_BATCH_SIZE:
                continue

            submit_batch(soma_dim_0_batch)
            soma_dim_0_batch = []

        # Process final batch
        if len(soma_dim_0_batch) > 0:
            submit_batch(soma_dim_0_batch)

        # Accumulate results

        n_cum_cells = 0
        for n, future in enumerate(futures.as_completed(batch_futures), start=1):
            result = future.result()
            # TODO: move writing of tiledb array to compute_all_estimators_for_batch_tdb; no need to return result
            if len(result) > 0:
                tiledb.from_pandas(ESTIMATORS_CUBE_ARRAY_URI, result.reset_index(CUBE_TILEDB_ATTRS_OBS), mode='append')
                logging.info("Pass 2: Writing to estimator cube.")
            else:
                logging.warning("Pass 2: Batch had empty result")
            logging.info(f"Pass 2: Completed {n} of {len(batch_futures)} batches ({100 * n / len(batch_futures):0.1f}%)")
            logging.debug(result)
            gc.collect()

        logging.info("Pass 2: Completed")


def run():
    # init multiprocessing
    start = time.time()
    if multiprocessing.get_start_method(True) != "spawn":
        multiprocessing.set_start_method("spawn", True)

    exp_uri = sys.argv[1] if len(sys.argv) > 1 else None
    layer = sys.argv[2] if len(sys.argv) > 2 else "raw"
    measurement_name = "RNA"

    with soma.Experiment.open(uri=exp_uri,
                              context=soma.SOMATileDBContext().replace(tiledb_config={
                                  "soma.init_buffer_bytes": TILEDB_SOMA_BUFFER_BYTES,
                                  "vfs.s3.region":"us-west-2",
                                  "vfs.s3.no_sign_request":True})
                              ) as exp:

        query = exp.axis_query(measurement_name=measurement_name,
                               obs_query=AxisQuery(value_filter=OBS_VALUE_FILTER_1),
                               # Note: Must use *all* genes to compute size factors correctly, even when var filter is
                               # being used for testing
                               var_query=AxisQuery())
        logging.info(f"Pass 1: Processing {query.n_obs} cells and {query.n_vars} genes")

        if not tiledb.array_exists(OBS_WITH_SIZE_FACTOR_TILEDB_ARRAY_URI):
            logging.info("Pass 1: Compute Approx Size Factors")
            size_factors = pass_1_compute_size_factors(query, layer)

            tiledb.from_pandas(OBS_WITH_SIZE_FACTOR_TILEDB_ARRAY_URI, size_factors.reset_index(), index_col=[0])
            logging.info("Saved `obs_with_size_factor` TileDB Array")
        else:
            logging.info("Pass 1: Compute Approx Size Factors (loading from stored data)")
            size_factors = tiledb.open(OBS_WITH_SIZE_FACTOR_TILEDB_ARRAY_URI).df[:].set_index('soma_joinid')

        logging.info("Pass 2: Compute Estimators")
        query = exp.axis_query(measurement_name=measurement_name,
                               obs_query=AxisQuery(value_filter=OBS_VALUE_FILTER_2),
                               var_query=AxisQuery(value_filter=VAR_VALUE_FILTER))
        logging.info(f"Pass 2: Processing {query.n_obs} cells and {query.n_vars} genes")

        pass_2_compute_estimators(query, size_factors, measurement_name=measurement_name, layer=layer)
        
        print('elapsed time', time.time()-start)

