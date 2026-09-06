"""Figure 2B (right) - recompute the Drop-seq gene-correlation estimates.

Port of publication/validation/estimation/smfish/correlation/correlation_estimation.py.

Why this has to be rerun rather than read off the data volume: the stored
smfish/correlation/sample_correlations.npz holds one unnamed column per gene pair,
in whatever order smfish_estimates.npz['corr_genes'] had at the time. That order
comes from a Python set intersection in preprocess_fish.py, so it is not stable
across runs, and the reference file on the volume was regenerated afterwards --
scoring the stored estimates against it yields negative correlations. Recomputing
against the current corr_genes restores the alignment.

SAVER and scVI estimates are read from their per-subsample csvs, which are indexed
by gene name and so realign on their own.
"""

import time

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.stats as stats

import config
import memento_size_factor

config.add_memento_oo_to_path()
import memento  # noqa: E402

SMFISH_PATH = config.FIGURE2_DATA + 'panelB_smfish/'
NUM_TRIALS = 20
NUMBER_OF_CELLS = [500, 1000, 5000, 8000]
MIN_MEAN_THRESH = 0.01
# Overall capture efficiency of the melanoma Drop-seq data, as used in the original script.
DROPSEQ_Q = 0.01485030176341905
METHODS = ['naive', 'saver', 'poisson', 'hypergeometric', 'scvi']


def load_pairs():
    dropseq_genes = sc.read_h5ad(SMFISH_PATH + 'reference/filtered_dropseq.h5ad').var.index.tolist()
    ref = np.load(SMFISH_PATH + 'reference/smfish_estimates.npz', allow_pickle=True)
    pairs = ref['corr_genes']
    idx1 = np.array([dropseq_genes.index(a) for a, _ in pairs])
    idx2 = np.array([dropseq_genes.index(b) for _, b in pairs])
    return dropseq_genes, pairs, idx1, idx2


def estimate(method, data, obs_mean, num_cell, trial, dropseq_genes, pairs, idx1, idx2):
    if method == 'naive':
        used = np.unique(np.concatenate([idx1, idx2]))
        position = {g: i for i, g in enumerate(used)}
        mat = data[:, used].toarray() / data.sum(axis=1).A1.reshape(-1, 1)
        values = np.array([stats.pearsonr(mat[:, position[a]], mat[:, position[b]])[0]
                           for a, b in zip(idx1, idx2)])
    elif method == 'poisson':
        sf = memento_size_factor.trimmed_size_factor(data, q=DROPSEQ_Q)
        values = memento.estimator.RNAPoisson().correlation(data, sf, idx1, idx2)
    elif method == 'hypergeometric':
        sf = memento_size_factor.trimmed_size_factor(data, q=DROPSEQ_Q)
        values = memento.estimator.RNAHypergeometric(DROPSEQ_Q).correlation(data, sf, idx1, idx2)
    elif method == 'saver':
        table = pd.read_csv(SMFISH_PATH + f'correlation/saver/{num_cell}_{trial}_corr2.csv', index_col=0)
        if num_cell < 8000:
            names = dict(zip([f'gene{i}' for i in range(len(dropseq_genes))], dropseq_genes))
            table.index = [names[n] for n in table.index]
            table.columns = table.index
        values = np.array([table.loc[a, b] for a, b in pairs])
    elif method == 'scvi':
        table = pd.read_csv(SMFISH_PATH + f'correlation/scvi/{num_cell}_{trial}_scvi_corr.csv', index_col=0)
        values = np.array([table.loc[a, b] for a, b in pairs])
    else:
        raise ValueError(f'Unknown method {method}')

    values = np.asarray(values, dtype=float)
    values[(obs_mean[idx1] < MIN_MEAN_THRESH) | (obs_mean[idx2] < MIN_MEAN_THRESH)] = np.nan
    return values


def main():
    dropseq_genes, pairs, idx1, idx2 = load_pairs()
    print(f'{len(pairs)} gene pairs', flush=True)

    estimates, details = [], []
    for num_cell in NUMBER_OF_CELLS:
        for trial in range(NUM_TRIALS if num_cell < 8000 else 1):
            start = time.time()
            adata = sc.read_h5ad(SMFISH_PATH + f'correlation/subsamples/{num_cell}_{trial}.h5ad')
            data = adata.X.tocsr()
            obs_mean = data.mean(axis=0).A1
            for method in METHODS:
                # SAVER and scVI were only run on a subset of the replicates.
                if method in ('saver', 'scvi') and trial > 0 and num_cell > 1000:
                    continue
                estimates.append(estimate(method, data, obs_mean, num_cell, trial,
                                          dropseq_genes, pairs, idx1, idx2))
                details.append((num_cell, trial + 1, method))
            print(f'{num_cell} cells, trial {trial}: {time.time() - start:.1f}s', flush=True)

    meta = pd.DataFrame(details, columns=['num_cell', 'trial', 'method'])
    np.savez_compressed(config.intermediate_path('panel_b_correlation_estimates.npz'),
                        correlations=np.vstack(estimates))
    meta.to_csv(config.intermediate_path('panel_b_correlation_metadata.csv'), index=False)
    print(f'wrote {len(estimates)} rows')


if __name__ == '__main__':
    main()
