"""Declarative inventory of the data files Figure 3 depends on.

Same contract as publication/figure2/data_manifest.py:

    python data_manifest.py check              # present on the source volume?
    python data_manifest.py check --root DIR   # ... or in an organized tree / bundle
    python data_manifest.py link               # build the tree as symlinks
    python data_manifest.py bundle --root DIR  # copy it into a standalone directory

Entries are tagged `required` (read directly by a panel script) or `provenance` (needed
only to regenerate a `required` file).
"""

import argparse
import os
import shutil

import config

REQUIRED, PROVENANCE = 'required', 'provenance'

CELL_TYPE = 'C'


def entries():
    """(panel, tier, destination in the organized tree, source path)."""
    out = []

    def add(panel, tier, dest, src):
        out.append((panel, tier, dest, config.DATA_PATH + src))

    # --- Panel A: UMAPs over the whole dataset -----------------------------
    add('A', REQUIRED, 'panelA_umap/HBEC_type_I_processed_deep.h5ad',
        'hbec/HBEC_type_I_processed_deep.h5ad')

    # --- Panels B and C: stored 1D tests per stim and timepoint ------------
    for stim in config.STIMS:
        for timepoint in config.TIMEPOINTS:
            name = f'{CELL_TYPE}_{stim}_{timepoint}.h5ad'
            add('BC', REQUIRED, f'panelBC_mean/tests/{name}', f'hbec/binary_test_latest/{name}')

    # --- Panels D-G: counts matrix, plus the tonic sensitivity source ------
    # The gene-by-gene correlations behind the ISG classification are computed from
    # the counts, so this h5ad is the upstream input for D, E, F and G alike.
    add('DEFG', REQUIRED, 'panelDEFG_isg/HBEC_type_I_filtered_counts_deep.h5ad',
        'hbec/HBEC_type_I_filtered_counts_deep.h5ad')
    add('DEFG', REQUIRED, 'panelDEFG_isg/external/mostafavi2016_mmc2.xls',
        'hbec/external/mostafavi2016_mmc2.xls')
    # The published DC table carries the `type` column from which the canonical and
    # non-canonical ISG lists are recovered; see isg_gene_lists.py.
    add('DEFG', REQUIRED, 'panelDEFG_isg/Supplementary_Table_2_HTEC_DC.csv',
        'tables/Supplementary_Table_2_HTEC_DC.csv')
    for stim in config.STIMS:
        name = f'{CELL_TYPE}_{stim}_6.h5ad'
        add('DEFG', REQUIRED, f'panelDEFG_isg/tests/{name}', f'hbec/binary_test_latest/{name}')

    # --- Provenance --------------------------------------------------------
    for stim in config.STIMS:
        name = f'{stim}_stratified_time.h5ad'
        add('BC', PROVENANCE, f'panelBC_mean/stratified/{name}',
            f'hbec/binary_test_stratified/{name}')
    add('DEFG', PROVENANCE, 'panelDEFG_isg/isg_classes/beta_ISGs.csv', 'hbec/isg_classes/beta_ISGs.csv')

    return out


def _resolve(row, root):
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
    for panel in ['A', 'BC', 'DEFG']:
        for tier in [REQUIRED, PROVENANCE]:
            subset = [r for r in rows if r[0] == panel and r[1] == tier]
            if not subset:
                continue
            present, missing, size = _summarize(subset, root)
            status = 'OK ' if not missing else 'GAP'
            print(f'{status} panel {panel:<5} {tier:<10} {len(present):>3}/{len(subset):<3} files'
                  f'  {size / 1e9:6.2f} GB')
            for row in missing[:5]:
                print(f'      missing: {_resolve(row, root)}')
            if missing and tier == REQUIRED:
                ok = False

    present, missing, size = _summarize(rows, root)
    print(f'\ntotal {len(present)}/{len(rows)} files, {size / 1e9:.2f} GB')
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
    parser.add_argument('--root', default=None)
    args = parser.parse_args()

    if args.command == 'check':
        raise SystemExit(0 if check(args.root) else 1)
    build(args.root or config.FIGURE3_DATA, copy=args.command == 'bundle')


if __name__ == '__main__':
    main()
