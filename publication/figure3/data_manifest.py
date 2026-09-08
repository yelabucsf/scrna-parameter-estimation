"""Declarative inventory of the data files Figure 3 depends on.

Same contract as publication/figure2/data_manifest.py:

    python data_manifest.py check                # present on the source volume?
    python data_manifest.py check --root DIR     # ... or in a downloaded bundle
    python data_manifest.py link                 # build the tree as symlinks
    python data_manifest.py bundle --root DIR    # copy it into a standalone directory
    python data_manifest.py archive --root DIR   # ... and pack it for publication

`bundle` and `archive` require --root: they write real copies, and defaulting to the
source volume would bury its symlink tree under gigabytes of duplicates.

This is maintainer tooling. Readers reproducing a figure download the published bundle
instead -- see publication/MAINTAINING.md.

Entries are tagged `required` (read directly by a panel script) or `provenance` (needed
only to regenerate a `required` file).
"""

import argparse
import os
import sys

import config

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bundle_tools  # noqa: E402

REQUIRED, PROVENANCE = bundle_tools.REQUIRED, bundle_tools.PROVENANCE
RESTRICTED = bundle_tools.RESTRICTED

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
    # Supplementary Table S1E of Mostafavi et al., Cell 2016 -- a publisher's
    # supplementary file. Tracked and linked locally so the panel runs here, but not
    # redistributed: reconstruct_tonic_isg.py prints the DOI to fetch it from.
    add('DEFG', RESTRICTED, 'panelDEFG_isg/external/mostafavi2016_mmc2.xls',
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
        for tier in [REQUIRED, PROVENANCE, RESTRICTED]:
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


def main():
    parser = bundle_tools.add_arguments(argparse.ArgumentParser(description=__doc__))
    bundle_tools.dispatch(parser.parse_args(), entries, check, config.FIGURE3_DATA)


if __name__ == '__main__':
    main()
