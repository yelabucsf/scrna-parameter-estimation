"""Declarative inventory of every data file Figure 2 depends on.

This is the single source of truth behind three subcommands:

    python data_manifest.py check     # is everything present?
    python data_manifest.py link      # build a panel-organized symlink tree
    python data_manifest.py bundle    # copy that tree into a portable directory

The volume at /memento_data is a flat sync of s3://memento-paper/revision/ and is
organized by *dataset*, which says nothing about which figure needs what. The tree
built here is organized by *panel and role* instead, so "what does panel B need" is
answerable by listing a directory.

Entries are tagged with a tier:
  required   - the panel scripts read this file directly
  provenance - not read at plot time, but needed to regenerate a `required` file
"""

import argparse
import os
import shutil

import config

REQUIRED, PROVENANCE = 'required', 'provenance'

CANOGAMEZ_DATASETS = ['CD4_Memory-Th0', 'CD4_Memory-Th2', 'CD4_Memory-Th17', 'CD4_Memory-iTreg',
                      'CD4_Naive-Th0', 'CD4_Naive-Th2', 'CD4_Naive-Th17', 'CD4_Naive-iTreg']
HAGAI_DATASETS = ['Hagai2018_mouse-lps', 'Hagai2018_mouse-pic', 'Hagai2018_pig-lps',
                  'Hagai2018_rabbit-lps', 'Hagai2018_rat-lps', 'Hagai2018_rat-pic']
BULK_METHODS = ['deseq2_lrt', 'deseq2_wald', 'edger_lrt', 'edger_qlft']
SC_METHODS = ['quasiGLM', 'edger_lrt', 'edger_qlft', 'deseq2_wald', 'deseq2_lrt', 't', 'MWU']

# The subsample sizes and replicate counts the smFISH scripts iterate over.
SMFISH_SIZES = [500, 1000, 5000, 8000]
SMFISH_TRIALS = 20
# BASiCS slice for panel A: the published panel is drawn at 100 cells and q < 0.6.
BASICS_NUM_CELL = 100
BASICS_CAPTURE_EFFICIENCIES = [0.05, 0.1, 0.2, 0.3, 0.5]
BASICS_TRIALS = 20


def _smfish_trials(num_cell, method_is_imputation=False):
    if num_cell >= 8000:
        return [0]
    if method_is_imputation and num_cell > 1000:
        return [0]
    return list(range(SMFISH_TRIALS))


def entries():
    """Yield (panel, tier, destination path in the organized tree, source path)."""
    out = []

    def add(panel, tier, dest, src):
        out.append((panel, tier, dest, config.DATA_PATH + src))

    # --- Panel A -----------------------------------------------------------
    add('A', REQUIRED, 'panelA_simulation/interferon_filtered.h5ad',
        'interferon_filtered.h5ad')
    for q in BASICS_CAPTURE_EFFICIENCIES:
        for trial in range(BASICS_TRIALS):
            name = f'{BASICS_NUM_CELL}_{q}_{trial}_parameters.csv'
            add('A', REQUIRED, f'panelA_simulation/basics/{name}', f'simulation/variance/{name}')

    # --- Panel B -----------------------------------------------------------
    add('B', REQUIRED, 'panelB_smfish/reference/smfish_estimates.npz',
        'smfish/smfish_estimates.npz')
    add('B', REQUIRED, 'panelB_smfish/reference/filtered_dropseq.h5ad',
        'smfish/filtered_dropseq.h5ad')
    add('B', PROVENANCE, 'panelB_smfish/reference/full_dropseq.h5ad', 'smfish/full_dropseq.h5ad')
    add('B', PROVENANCE, 'panelB_smfish/reference/fishSubset.txt', 'smfish/fishSubset.txt')

    for name in ['sample_means.npz', 'sample_metadata.csv']:
        add('B', REQUIRED, f'panelB_smfish/mean/{name}', f'smfish/mean/{name}')
    for name in ['sample_variances.npz', 'sample_means.npz', 'sample_metadata.csv']:
        add('B', REQUIRED, f'panelB_smfish/variance/{name}', f'smfish/variance/{name}')

    for num_cell in SMFISH_SIZES:
        for trial in _smfish_trials(num_cell):
            name = f'{num_cell}_{trial}.h5ad'
            add('B', REQUIRED, f'panelB_smfish/correlation/subsamples/{name}',
                f'smfish/variance/{name}')
        # BASiCS, SAVER and scVI were each only run on trial 0 above 1000 cells.
        for trial in _smfish_trials(num_cell, method_is_imputation=True):
            add('B', PROVENANCE, f'panelB_smfish/variance/basics/{num_cell}_{trial}_parameters.csv',
                f'smfish/variance/{num_cell}_{trial}_parameters.csv')
            add('B', REQUIRED, f'panelB_smfish/correlation/saver/{num_cell}_{trial}_corr2.csv',
                f'smfish/correlation/{num_cell}_{trial}_corr2.csv')
            add('B', REQUIRED, f'panelB_smfish/correlation/scvi/{num_cell}_{trial}_scvi_corr.csv',
                f'smfish/correlation/{num_cell}_{trial}_scvi_corr.csv')

    # --- Panel C -----------------------------------------------------------
    for name in ['anndata.h5ad', 'memento_wls.csv', 'edger_lrt.csv', 'edger_qlft.csv', 't.csv']:
        add('C', REQUIRED, f'panelC_inference/dm/{name}', f'simulation/de/{name}')
    for name in ['high_expr_anndata.h5ad', 'memento.csv', 'dv_basics.csv']:
        add('C', REQUIRED, f'panelC_inference/dv/{name}', f'simulation/dv/{name}')
    for name in ['high_expr_anndata_clean.h5ad', 'obs.csv']:
        add('C', PROVENANCE, f'panelC_inference/dv/{name}', f'simulation/dv/{name}')
    for name in ['memento_dc.csv', 'dc_schot.csv', 'dc_true_effect_size.pkl']:
        add('C', REQUIRED, f'panelC_inference/dc/{name}', f'simulation/dc/{name}')

    # --- Panel D -----------------------------------------------------------
    for dataset in CANOGAMEZ_DATASETS:
        for method in BULK_METHODS:
            name = f'{dataset}_{method}.csv'
            add('D', REQUIRED, f'panelD_bulk/canogamez/bulk/{name}',
                f'canogamez/bulk_results/{name}')
        for method in SC_METHODS:
            name = f'{dataset}_1_{method}.csv'
            add('D', REQUIRED, f'panelD_bulk/canogamez/single_cell/{name}',
                f'canogamez/sc_results/{name}')
    for dataset in HAGAI_DATASETS:
        for method in BULK_METHODS:
            name = f'{dataset}_{method}.csv'
            add('D', REQUIRED, f'panelD_bulk/hagai/bulk/{name}',
                f'hagai/bulk_rnaseq/results/{name}')
        for method in SC_METHODS:
            name = f'{dataset}_{method}.csv'
            add('D', REQUIRED, f'panelD_bulk/hagai/single_cell/{name}',
                f'hagai/sc_rnaseq/results/{name}')
    for trial in range(50):
        for name in [f'T4_vs_cM.bulk.edger_lrt.100.{trial}.csv',
                     f'T4_vs_cM.bulk.edger_qlft.100.{trial}.csv',
                     f'T4_vs_cM.bulk.deseq2_wald.100.{trial}.csv',
                     f'T4_vs_cM.bulk.deseq2_lrt.100.{trial}.csv']:
            add('D', REQUIRED, f'panelD_bulk/lupus/bulk/{name}', f'lupus_bulk/{name}')
        for name in [f'T4_vs_cM.pseudobulk.edger_lrt.100.{trial}.csv',
                     f'T4_vs_cM.pseudobulk.deseq2_wald.100.{trial}.csv',
                     f'100_{trial}_quasiGLM.csv', f'100_{trial}_t.csv', f'100_{trial}_mwu.csv']:
            add('D', REQUIRED, f'panelD_bulk/lupus/single_cell/{name}', f'lupus_bulk/{name}')

    return out


def _resolve(row, root):
    """Path to test: inside an organized tree if given a root, else the source volume."""
    return os.path.join(root, row[2]) if root else row[3]


def _summarize(rows, root):
    present = [r for r in rows if os.path.exists(_resolve(r, root))]
    missing = [r for r in rows if not os.path.exists(_resolve(r, root))]
    size = sum(os.path.getsize(_resolve(r, root)) for r in present)
    return present, missing, size


def check(root=None):
    rows = entries()
    ok = True
    print(f'checking {root or config.DATA_PATH} '
          f'({"organized tree" if root else "source volume"})\n')
    for panel in ['A', 'B', 'C', 'D', 'E']:
        for tier in [REQUIRED, PROVENANCE]:
            subset = [r for r in rows if r[0] == panel and r[1] == tier]
            if not subset:
                continue
            present, missing, size = _summarize(subset, root)
            status = 'OK ' if not missing else 'GAP'
            print(f'{status} panel {panel} {tier:<10} {len(present):>4}/{len(subset):<4} files'
                  f'  {size / 1e9:6.2f} GB')
            for row in missing[:5]:
                print(f'      missing: {_resolve(row, root)}')
            if len(missing) > 5:
                print(f'      ... and {len(missing) - 5} more')
            if missing and tier == REQUIRED:
                ok = False

    present, missing, size = _summarize(rows, root)
    print(f'\ntotal {len(present)}/{len(rows)} files, {size / 1e9:.2f} GB')
    print('Panel E needs no data files; its runtimes are literals in panel_e_runtime.py.')
    return ok


def build(root, copy):
    rows = [r for r in entries() if os.path.exists(r[3])]
    for _, _, dest, src in rows:
        target = os.path.join(root, dest)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        if os.path.lexists(target):
            os.remove(target)
        if copy:
            shutil.copy2(src, target)
        else:
            os.symlink(os.path.realpath(src), target)
    print(f'{"copied" if copy else "linked"} {len(rows)} files into {root}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['check', 'link', 'bundle'])
    parser.add_argument('--root', default=None,
                        help='organized tree to build, or to check instead of the source volume')
    args = parser.parse_args()

    if args.command == 'check':
        raise SystemExit(0 if check(args.root) else 1)
    build(args.root or config.FIGURE2_DATA, copy=args.command == 'bundle')


if __name__ == '__main__':
    main()
