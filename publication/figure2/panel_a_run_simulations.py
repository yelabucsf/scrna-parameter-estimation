"""Figure 2A - run the estimator simulations.

Ports publication/validation/estimation/simulation/{mean,variance,correlation}/*.py
onto the current data layout. The variance simulation originally round-tripped every
replicate through h5ad files so that BASiCS could be run in R; here the estimates are
computed in memory. BASiCS is therefore not part of the regenerated variance curve.

Writes one npz of estimates plus one csv of metadata per quantity into intermediate/.
"""

import argparse
import itertools
import os
import time

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.io
import scipy.sparse as sparse
import sklearn.datasets as sklearn_datasets

import config

config.add_memento_oo_to_path()
import memento  # noqa: E402
import memento.auxillary.simulate as simulate  # noqa: E402


CELL_TYPE = 'CD4 T cells - ctrl'
REFERENCE_Q = 0.07
MIN_MEAN = 0.01
NUM_CORR_GENES = 300

CAPTURE_EFFICIENCIES = {
    'mean': [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1],
    'variance': [0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1],
    'correlation': [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1],
}
NUMBER_OF_CELLS = {
    'mean': [10, 500],
    'variance': [50, 100, 500],
    'correlation': [50, 100, 500],
}
METHODS = {
    'mean': ['naive', 'pb', 'hypergeometric'],
    'variance': ['ground_truth', 'naive', 'poisson', 'hypergeometric'],
    'correlation': ['ground_truth', 'naive', 'poisson', 'hypergeometric'],
}

# The slice of the variance simulation that run_basics_simulation.R scores, matching
# the (num_cell, q) the published panel is drawn at. Dumped as MatrixMarket so R can
# read the counts without Seurat/SeuratDisk.
#
# This is a regeneration output, so it goes to the local intermediate directory rather
# than to the data bundle: the panel itself is drawn from the BASiCS results already in
# the bundle, and someone re-running the simulation should not be writing back into their
# downloaded inputs. Override with FIGURE2_BASICS_DIR.
BASICS_DUMP_DIR = os.environ.get(
    'FIGURE2_BASICS_DIR', os.path.join(config.INTERMEDIATE_DIR, 'basics_simulation')) + '/'
BASICS_NUM_CELL = 100
BASICS_CAPTURE_EFFICIENCIES = [0.05, 0.1, 0.2, 0.3, 0.5]


def get_simulation_parameters(q=REFERENCE_Q):
    adata = sc.read(config.FIGURE2_DATA + 'panelA_simulation/interferon_filtered.h5ad')
    adata = adata[adata.obs.cell_type == CELL_TYPE]
    x_param, z_param, Nc, _ = simulate.extract_parameters(adata.X, q=q, min_mean=MIN_MEAN)
    return x_param, z_param, Nc


def simulate_data(n_cells, q, z_param, Nc, cov_matrix='uncorrelated'):
    true_data = simulate.simulate_transcriptomes(
        n_cells=n_cells, means=z_param[0], variances=z_param[1], Nc=Nc, norm_cov=cov_matrix)
    true_data[true_data < 0] = 0
    _, captured_data = simulate.capture_sampling(true_data, q, q_sq=None)
    return true_data, captured_data


def estimate_mean(captured_data, size_factor, method, q):
    if method == 'naive':
        return (captured_data / size_factor.reshape(-1, 1)).mean(axis=0).A1
    if method == 'pb':
        return captured_data.sum(axis=0).A1 / captured_data.sum()
    if method == 'hypergeometric':
        return memento.estimator.RNAHypergeometric(q).mean(captured_data, size_factor)
    raise ValueError(f'Unknown mean method {method}')


def estimate_variance(true_data, captured_data, size_factor, method, q):
    if method == 'ground_truth':
        true_size_factor = true_data.sum(axis=1).A1
        return (true_data.toarray() / true_size_factor.reshape(-1, 1)).var(axis=0)
    if method == 'naive':
        return (captured_data.toarray() / size_factor.reshape(-1, 1)).var(axis=0)
    if method == 'poisson':
        return memento.estimator.RNAPoisson().variance(captured_data, size_factor)
    if method == 'hypergeometric':
        return memento.estimator.RNAHypergeometric(q).variance(captured_data, size_factor)
    raise ValueError(f'Unknown variance method {method}')


def estimate_correlation(true_data, captured_data, method, q, num_genes, idx1, idx2):
    if method == 'ground_truth':
        size_factor = true_data.sum(axis=1).A1
        relative = true_data[:, :num_genes].toarray() / size_factor.reshape(-1, 1)
        mat = np.corrcoef(relative, rowvar=False)
        return mat[idx1, idx2]
    if method == 'naive':
        size_factor = captured_data.sum(axis=1).A1
        relative = captured_data[:, :num_genes].toarray() / size_factor.reshape(-1, 1)
        mat = np.corrcoef(relative, rowvar=False)
        return mat[idx1, idx2]
    size_factor = captured_data.sum(axis=1).A1
    if method == 'poisson':
        return memento.estimator.RNAPoisson().correlation(captured_data, size_factor, idx1, idx2)
    if method == 'hypergeometric':
        return memento.estimator.RNAHypergeometric(q).correlation(captured_data, size_factor, idx1, idx2)
    raise ValueError(f'Unknown correlation method {method}')


def run_mean(z_param, Nc, x_param, num_trials, rng):
    num_genes = x_param[0].shape[0]
    rows, details = [x_param[0]], [(1, np.inf, 0, 'ground_truth')]
    for q in CAPTURE_EFFICIENCIES['mean']:
        for num_cell in NUMBER_OF_CELLS['mean']:
            for trial in range(num_trials):
                _, captured = simulate_data(num_cell, q, z_param, Nc)
                captured = sparse.csr_matrix(captured)
                size_factor = captured.sum(axis=1).A1
                for method in METHODS['mean']:
                    rows.append(estimate_mean(captured, size_factor, method, q))
                    details.append((q, num_cell, trial + 1, method))
    meta = pd.DataFrame(details, columns=['q', 'num_cell', 'trial', 'method'])
    return np.vstack(rows).reshape(-1, num_genes), meta


def dump_for_basics(captured, q, num_cell, trial):
    """Write one replicate as genes x cells MatrixMarket, plus the retained gene indices.

    BASiCS cannot fit genes that are zero in every cell, so those are dropped here and
    the surviving indices recorded, letting the Python side realign the results.
    """
    os.makedirs(BASICS_DUMP_DIR, exist_ok=True)
    kept = np.where(captured.sum(axis=0).A1 > 0)[0]
    name = f'{num_cell}_{q}_{trial}'
    scipy.io.mmwrite(BASICS_DUMP_DIR + f'{name}_counts.mtx', captured[:, kept].T.astype(int))
    pd.Series(kept, name='gene_index').to_csv(BASICS_DUMP_DIR + f'{name}_genes.csv', index=False)


def run_variance(z_param, Nc, num_trials, rng, dump_basics=False):
    rows, details = [], []
    for q in CAPTURE_EFFICIENCIES['variance']:
        for num_cell in NUMBER_OF_CELLS['variance']:
            for trial in range(num_trials):
                true_data, captured = simulate_data(num_cell, q, z_param, Nc)
                true_data = sparse.csr_matrix(true_data)
                captured = sparse.csr_matrix(captured)
                size_factor = captured.sum(axis=1).A1
                if (dump_basics and num_cell == BASICS_NUM_CELL
                        and q in BASICS_CAPTURE_EFFICIENCIES):
                    dump_for_basics(captured, q, num_cell, trial)
                for method in METHODS['variance']:
                    rows.append(estimate_variance(true_data, captured, size_factor, method, q))
                    details.append((q, num_cell, trial + 1, method))
    meta = pd.DataFrame(details, columns=['q', 'num_cell', 'trial', 'method'])
    return np.vstack(rows), meta


def run_correlation(z_param, Nc, num_trials, rng):
    num_pairs = NUM_CORR_GENES * (NUM_CORR_GENES - 1) // 2
    estimates = []
    details = []
    for q in CAPTURE_EFFICIENCIES['correlation']:
        for num_cell in NUMBER_OF_CELLS['correlation']:
            for trial in range(num_trials):
                cov_matrix = sklearn_datasets.make_spd_matrix(
                    NUM_CORR_GENES, random_state=rng.integers(2 ** 31))
                true_data, captured = simulate_data(num_cell, q, z_param, Nc, cov_matrix=cov_matrix)

                cell_filter = captured.sum(axis=1) > 0
                gene_filter = captured.sum(axis=0) > 0
                true_data = sparse.csr_matrix(true_data[cell_filter, :][:, gene_filter])
                captured = sparse.csr_matrix(captured[cell_filter, :][:, gene_filter])
                num_genes = int(gene_filter[:NUM_CORR_GENES].sum())

                idx1, idx2 = map(np.array, zip(*itertools.combinations(np.arange(num_genes), 2)))
                num_avail = idx1.shape[0]
                for method in METHODS['correlation']:
                    row = np.full(num_pairs, np.nan)
                    row[:num_avail] = estimate_correlation(
                        true_data, captured, method, q, num_genes, idx1, idx2)
                    estimates.append(row)
                    details.append((q, num_cell, trial + 1, method, num_avail))
    meta = pd.DataFrame(details, columns=['q', 'num_cell', 'trial', 'method', 'num_pairs'])
    return np.vstack(estimates), meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('quantity', choices=['mean', 'variance', 'correlation', 'all'])
    parser.add_argument('--num-trials', type=int, default=20)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dump-basics-inputs', action='store_true',
                        help='write the variance replicates that run_basics_simulation.R scores')
    args = parser.parse_args()

    x_param, z_param, Nc = get_simulation_parameters()
    print(f'simulation parameters: {x_param[0].shape[0]} genes, mean Nc {np.mean(Nc):.0f}', flush=True)

    quantities = ['mean', 'variance', 'correlation'] if args.quantity == 'all' else [args.quantity]

    for offset, quantity in enumerate(['mean', 'variance', 'correlation']):
        if quantity not in quantities:
            continue
        # Seed per quantity so that running one alone reproduces what 'all' produces.
        rng = np.random.default_rng(args.seed + offset)
        np.random.seed(args.seed + offset)

        start = time.time()
        if quantity == 'mean':
            estimates, meta = run_mean(z_param, Nc, x_param, args.num_trials, rng)
        elif quantity == 'variance':
            estimates, meta = run_variance(z_param, Nc, args.num_trials, rng,
                                           dump_basics=args.dump_basics_inputs)
        else:
            estimates, meta = run_correlation(z_param, Nc, args.num_trials, rng)
        np.savez_compressed(config.intermediate_path(f'panel_a_{quantity}_estimates.npz'),
                            estimates=estimates)
        meta.to_csv(config.intermediate_path(f'panel_a_{quantity}_metadata.csv'), index=False)
        print(f'{quantity}: {estimates.shape} in {time.time() - start:.0f}s', flush=True)


if __name__ == '__main__':
    main()
