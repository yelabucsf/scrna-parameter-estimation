"""Declarative inventory of the data files Figure 5 depends on.

Same contract as publication/figure{2,3,4}/data_manifest.py:

    python data_manifest.py check                # present on the source volume?
    python data_manifest.py check --root DIR     # ... or in a downloaded bundle
    python data_manifest.py link                 # build the tree as symlinks
    python data_manifest.py bundle --root DIR    # copy it into a standalone directory
    python data_manifest.py archive --root DIR   # ... and pack it for publication

`bundle` and `archive` require --root: they write real copies, and defaulting to the
source volume would bury its symlink tree under gigabytes of duplicates.

This is maintainer tooling. Readers reproducing a figure download the published bundle
instead -- see publication/MAINTAINING.md.

Entries are tagged `required` (read directly by a panel script), `provenance` (needed only
to regenerate a `required` file), or `restricted` (may not be redistributed -- tracked and
linked locally, never copied into a bundle).
"""

import argparse
import os
import sys

import config

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bundle_tools  # noqa: E402

REQUIRED, PROVENANCE = bundle_tools.REQUIRED, bundle_tools.PROVENANCE
RESTRICTED = bundle_tools.RESTRICTED


def entries():
    """(panel, tier, destination in the organized tree, source path)."""
    out = []

    def add(panel, tier, dest, src):
        out.append((panel, tier, dest, config.LUPUS_PATH + src))

    # Genotypes. Individual-level data from the CLUES cohort, released by Perez et al.
    # 2022 only under dbGaP phs002812.v1.p1 with a signed Data Use Certification, so they
    # carry the RESTRICTED tier and are excluded from any published bundle.
    #
    # Panel A does not actually need them: it applies a minor-allele-frequency filter,
    # which is one aggregate number per variant. panel_a_qqplots.write_minor_allele_
    # frequencies() reduces the matrices to exactly that, and the aggregate is what ships.
    # Panels F-I do need per-individual calls and are skipped without dbGaP access.
    for population in config.POPULATIONS:
        add('A', RESTRICTED, f'genotypes/{population}_genos.tsv',
            f'mateqtl_input/{population}_genos.tsv')
        add('A', REQUIRED, f'panelA_qq/maf/{population}_maf.csv',
            f'mateqtl_input/maf/{population}_maf.csv')

    # --- Panel A: QQ plots for eQTL, vQTL and cQTL ---------------------------
    for population in config.POPULATIONS:
        for cell_type in config.CELL_TYPES:
            stem = f'{population}_{cell_type}'
            add('A', REQUIRED, f'panelA_qq/memento/{stem}.csv',
                f'full_analysis/memento/100kb/{stem}.csv')
            add('A', REQUIRED, f'panelA_qq/memento/{stem}_variability.csv',
                f'full_analysis/memento/100kb/{stem}_variability.csv')
            add('A', REQUIRED, f'panelA_qq/memento/{stem}_coexpression.csv',
                f'full_analysis/memento/100kb/{stem}_coexpression.csv')
            add('A', REQUIRED, f'panelA_qq/mateqtl/{stem}_all_hg19.csv',
                f'full_analysis/mateqtl/outputs/{stem}_all_hg19.csv')

    # --- Panels B and C: replication against OneK1K, and power vs cohort size --
    add('BC', REQUIRED, 'panelBC_replication/filtered_onek_eqtls.csv',
        'filtered_onek_eqtls.csv')
    add('BC', PROVENANCE, 'panelBC_replication/OneK1K_eqtls_for_replication.txt',
        'OneK1K_eqtls_for_replication.txt')
    # Full-cohort results plus permuted-genotype nulls, for the panel B ROC.
    for cell_type in config.CELL_TYPES:
        add('BC', REQUIRED, f'panelBC_replication/memento_1k/asian_{cell_type}.csv',
            f'memento_1k/asian_{cell_type}.csv')
        add('BC', REQUIRED, f'panelBC_replication/memento_1k/{cell_type}_shuffled.csv',
            f'memento_1k/{cell_type}_shuffled.csv')
        add('BC', REQUIRED,
            f'panelBC_replication/mateqtl_filtered/asian_{cell_type}_filtered.out',
            f'mateqtl_output/asian_{cell_type}_filtered.out')
        add('BC', REQUIRED,
            f'panelBC_replication/mateqtl_filtered/{cell_type}_filtered_shuffled.out',
            f'mateqtl_output/{cell_type}_filtered_shuffled.out')

    # Subsampled cohorts: 10 resamples per size, both methods, six cell types.
    for size in [50, 60, 70, 80]:
        for resample in range(10):
            for cell_type in config.CELL_TYPES:
                stem = f'asian_{cell_type}_{size}_{resample}'
                add('BC', REQUIRED, f'panelBC_replication/memento_1k/{stem}.csv',
                    f'memento_1k/{stem}.csv')
                add('BC', REQUIRED, f'panelBC_replication/mateqtl_sampled/{stem}.out',
                    f'mateqtl_output/sampled/{stem}.out')

    # --- Panels D and E: eQTL enrichment in cell-type-specific ATAC peaks -----
    add('DE', REQUIRED, 'panelDE_atac/sorted_simple_atac_lineage_groups3.bed.gz',
        'atac_enrichment/sorted_simple_atac_lineage_groups3.bed.gz')

    for population in config.POPULATIONS:
        for cell_type in config.CELL_TYPES:
            for method in ['memento', 'matqetl']:
                name = f'{population}_{method}_{cell_type}.out'
                add('DE', REQUIRED, f'panelDE_atac/enrichment/{name}',
                    f'atac_enrichment/100kb/{name}')
    for population in config.POPULATIONS:
        for cell_type, group in [('B', 'B'), ('T4', 'T'), ('T8', 'T'), ('T8', 'nk'),
                                 ('NK', 'nk'), ('cM', 'myeloid'), ('ncM', 'myeloid')]:
            for method in ['memento', 'mateqtl']:
                name = f'{population}_{cell_type}_{group}.txt'
                add('DE', REQUIRED, f'panelDE_atac/peaks/{method}/{name}',
                    f'atac_enrichment/peaks/{method}/{name}')

    # --- Panels F-I: the vQTL and cQTL worked examples ------------------------
    for population in config.POPULATIONS:
        for cell_type in config.CELL_TYPES:
            add('FI', REQUIRED, f'panelFI_examples/single_cell/{population}_{cell_type}.h5ad',
                f'single_cell/{population}_{cell_type}.h5ad')
    add('FI', REQUIRED, 'panelFI_examples/Supplementary_Table_6_SLE_vQTL.csv',
        '../tables/Supplementary_Table_6_SLE_vQTL.csv')
    add('FI', REQUIRED, 'panelFI_examples/Supplementary_Table_7_SLE_cQTL.csv',
        '../tables/Supplementary_Table_7_SLE_cQTL.csv')

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
    for panel in ['A', 'BC', 'DE', 'FI']:
        for tier in [REQUIRED, PROVENANCE, RESTRICTED]:
            subset = [r for r in rows if r[0] == panel and r[1] == tier]
            if not subset:
                continue
            present, missing, size = _summarize(subset, root)
            # Restricted files are absent from a published bundle by design, so their
            # absence is the expected state rather than a gap.
            status = 'OK ' if not missing else ('--- ' if tier == RESTRICTED else 'GAP')
            print(f'{status} panel {panel:<3} {tier:<10} {len(present):>2}/{len(subset):<2} files'
                  f'  {size / 1e9:6.2f} GB')
            if tier == RESTRICTED and missing:
                print('      withheld: dbGaP phs002812; panels F-I are skipped without it')
            else:
                for row in missing[:5]:
                    print(f'      missing: {_resolve(row, root)}')
            if missing and tier == REQUIRED:
                ok = False

    present, missing, size = _summarize(rows, root)
    print(f'\ntotal {len(present)}/{len(rows)} files, {size / 1e9:.2f} GB')
    return ok


def main():
    parser = bundle_tools.add_arguments(argparse.ArgumentParser(description=__doc__))
    bundle_tools.dispatch(parser.parse_args(), entries, check, config.FIGURE5_DATA)


if __name__ == '__main__':
    main()
