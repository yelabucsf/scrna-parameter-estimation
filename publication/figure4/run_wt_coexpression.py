"""Gene-by-gene memento correlations among the DMGs, in wild-type cells.

Port of publication/original/perturbseq/cd4_wt_coex.ipynb cells 42-47, the right half of panel 4C.

This has to be computed rather than read off the volume: `2d/wt_one_sample.csv` holds
10,721 regulator-target pairs, only 563 of the 64,261 DMG-DMG pairs the panel needs.
"""

import itertools
import time

import numpy as np
import pandas as pd
import scanpy as sc

import config
import perturbseq_data

config.add_repo_to_path()
import memento  # noqa: E402

# Overall capture efficiency for this experiment, from the notebook.
CAPTURE_EFFICIENCY = 0.11


def main():
    genes = perturbseq_data.dmg_set()
    print(f'{len(genes)} DMGs', flush=True)

    adata = sc.read(perturbseq_data.COUNTS)
    adata.obs['q'] = CAPTURE_EFFICIENCY
    memento.setup_memento(adata, q_column='q', trim_percent=0.1)

    wild_type = adata[adata.obs['WT'] == 'T'].copy()
    wild_type.obs['group'] = 'A'
    print(f'{wild_type.shape[0]} wild-type cells', flush=True)

    memento.create_groups(wild_type, label_columns=['group'])
    memento.compute_1d_moments(wild_type, gene_list=genes)

    present = [g for g in genes if g in wild_type.var.index]
    pairs = list(itertools.combinations(present, 2))
    print(f'{len(present)} genes survive the moment filters, {len(pairs)} pairs', flush=True)

    start = time.time()
    memento.compute_2d_moments(wild_type, pairs)
    moments = memento.get_2d_moments(wild_type, groupby='group')
    print(f'2d moments in {time.time() - start:.0f}s', flush=True)

    matrix = pd.DataFrame(np.nan, index=present, columns=present)
    for _, row in moments.iterrows():
        matrix.loc[row['gene_1'], row['gene_2']] = row['group_A']
        matrix.loc[row['gene_2'], row['gene_1']] = row['group_A']

    out = config.intermediate_path('wt_dmg_coexpression.csv')
    matrix.to_csv(out)
    print(f'wrote {out} {matrix.shape}')


if __name__ == '__main__':
    main()
