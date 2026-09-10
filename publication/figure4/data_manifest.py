"""Declarative inventory of the data files Figure 4 depends on.

Same contract as publication/figure{2,3}/data_manifest.py:

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


# The 84 sgRNAs that survive perturbseq_data.selected_guides(), frozen here because the
# manifest must be able to enumerate itself *before* the tree it describes exists --
# selected_guides() reads tfko.sng.guides.full.ct.h5ad out of FIGURE4_DATA, which is
# circular when checking a fresh download or building a bundle from scratch.
# perturbseq_data.selected_guides() remains the source of truth for the analysis; this is
# a cached copy of its output, and verify_guides() below asserts the two still agree.
TESTED_GUIDES = [
    'ATF3.212615192', 'ATF4.39521667', 'ATF4.39521890', 'ATF6.161846504',
    'BACH1.29321463', 'BACH1.29326546', 'BHLHE40.4980001', 'CREM.35188237',
    'CTCFL.57515801', 'CTCFL.57518862', 'DNMT1.10146475', 'DPF2.65340430',
    'EGR1.138467317', 'EGR1.138467471', 'EGR2.62815887', 'ELK1.47638134',
    'ELK4.205623693', 'ERG.38392383', 'ERG.38445525', 'EWSR1.29282556',
    'EWSR1.29288709', 'EZH2.148826587', 'FOSL2.28408792', 'FOSL2.28412051',
    'FOXP1.71015617', 'FUBP1.77964146', 'FUBP1.77969992', 'FUS.31183999',
    'FUS.31184329', 'GABPA.25741671', 'GABPB1.50303991', 'GABPB1.50309696',
    'GTF2I.74699069', 'HCFC1.153963360', 'HDAC3.141634859', 'IFI16.159018299',
    'IKZF1.50376659', 'IRF1.132487047', 'IRF1.132487119', 'IRF2.184418577',
    'IRF4.394977', 'KLF6.3782035', 'MAF1.144106162', 'MAFK.1540041', 'MATR3.139307613',
    'MIER1.66958196', 'MIER1.66970832', 'MLX.42569204', 'MTA2.62596475',
    'MTA2.62598030', 'NCOA3.47627735', 'NCOA3.47634078', 'NCOA4.46012894',
    'NFATC3.68122059', 'NFATC3.68183267', 'NONO.71294280', 'NONO.71296973',
    'NRF1.129710492', 'PCBP2.53455366', 'PCBP2.53459399', 'PHB.49411683',
    'PHB.49411797', 'PHB2.6969554', 'POLR2A.7498096', 'PRDM1.106088300',
    'PRDM1.106105284', 'SLC30A9.42063117', 'SMAD2.47870480', 'SMAD2.47896526',
    'SMARCA5.143536626', 'SP1.53383311', 'SSRP1.57331762', 'STAT1.190997935',
    'TAF7.141319508', 'TFDP1.113633911', 'TOE1.45342886', 'TP53.7675058',
    'YBX1.42696671', 'ZNF146.36236488', 'ZNF207.32351892', 'ZNF24.35339842',
    'ZNF24.35340244', 'ZNF460.57291533', 'ZNF622.16463242',
]


def _tested_guides():
    return TESTED_GUIDES


def verify_guides():
    """Recompute the guide list from the h5ad and compare against the frozen copy.

    Needs the data, so it is not part of `check`; run it after changing the selection
    filters in perturbseq_data.py.
    """
    import perturbseq_data
    live = perturbseq_data.selected_guides()
    if live != TESTED_GUIDES:
        missing = sorted(set(live) - set(TESTED_GUIDES))
        extra = sorted(set(TESTED_GUIDES) - set(live))
        raise SystemExit(
            f'TESTED_GUIDES is stale: {len(live)} live vs {len(TESTED_GUIDES)} frozen.\n'
            f'  in selected_guides() but not frozen: {missing}\n'
            f'  frozen but no longer selected:       {extra}')
    print(f'TESTED_GUIDES matches selected_guides() ({len(live)} sgRNAs)')


def entries():
    """(panel, tier, destination in the organized tree, source path)."""
    out = []

    def add(panel, tier, dest, src, root=None):
        out.append((panel, tier, dest, (root or config.DATA_PATH) + src))

    # --- Panel A: which regulators were perturbed, by expression and binding ---
    add('A', REQUIRED, 'panelA_selection/tfko.sng.guides.full.ct.h5ad',
        'tfko140/tfko.sng.guides.full.ct.h5ad')
    add('A', REQUIRED, 'panelA_selection/encode_result.csv', 'tfko140/encode_result.csv')

    # --- Panels B, C, D: differential mean per sgRNA, and WT coexpression ---
    add('BCD', REQUIRED, 'panelBCD_effects/filtered_1d_result.csv',
        'tfko140/1d/filtered_1d_result.csv')
    add('BCD', REQUIRED, 'panelBCD_effects/wt_one_sample.csv', 'tfko140/2d/wt_one_sample.csv')
    add('BCD', PROVENANCE, 'panelBCD_effects/raw_1d_result.csv', 'tfko140/1d/raw_1d_result.csv')

    # --- Panels E, F: the two regulatory networks -----------------------------
    # cytoscape_SIF*.csv are the edge lists the published networks were laid out from.
    add('EF', REQUIRED, 'panelEF_network/guide_combine_donor.csv',
        'tfko140/2d/guide_combine_donor.csv')
    add('EF', REQUIRED, 'panelEF_network/cytoscape_SIF.csv', 'tfko140/cytoscape_SIF.csv')
    add('EF', REQUIRED, 'panelEF_network/cytoscape_SIF_explicit.csv',
        'tfko140/cytoscape_SIF_explicit.csv')
    add('EF', REQUIRED, 'panelEF_network/Supplementary_Table_3_Perturb-seq_DM.csv',
        'tables/Supplementary_Table_3_Perturb-seq_DM.csv')
    add('EF', REQUIRED, 'panelEF_network/Supplementary_Table_4_Perturb-seq_DC.csv',
        'tables/Supplementary_Table_4_Perturb-seq_DC.csv')

    # Per-guide differential correlation tests, Fisher-combined per regulator to decide
    # which regulator pairs interact (panel G's split).
    for guide in _tested_guides():
        name = f'{guide}_vs_WT.csv'
        add('GH', REQUIRED, f'panelGH_chipseq/dc_tests/{name}', f'tfko140/2d_tests/{name}')

    # --- Panels G, H: ChIP-seq binding relative to the TSS --------------------
    add('GH', REQUIRED, 'panelGH_chipseq/encode_result.csv', 'tfko140/encode_result.csv')
    add('GH', REQUIRED, 'panelGH_chipseq/GRCh38Genes.bed', 'GRCh38Genes.bed',
        root=config.MISCSEQ_PATH + '/')
    # ENCODE peak sets for the panel H locus view, selected by the same rule the
    # notebook's Encode helper used (it streamed and deleted them, leaving nothing).
    for accession in ['ENCFF557FUM', 'ENCFF719BHI']:
        add('GH', REQUIRED, f'panelGH_chipseq/peaks/{accession}.bed.gz',
            f'tfko140/encode_peaks/{accession}.bed.gz')
    add('GH', PROVENANCE, 'panelGH_chipseq/activator_interaction.h5ad',
        'tfko140/activator_interaction.h5ad')

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
    for panel in ['A', 'BCD', 'EF', 'GH']:
        for tier in [REQUIRED, PROVENANCE]:
            subset = [r for r in rows if r[0] == panel and r[1] == tier]
            if not subset:
                continue
            present, missing, size = _summarize(subset, root)
            status = 'OK ' if not missing else 'GAP'
            print(f'{status} panel {panel:<4} {tier:<10} {len(present):>2}/{len(subset):<2} files'
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
    bundle_tools.dispatch(parser.parse_args(), entries, check, config.FIGURE4_DATA)


if __name__ == '__main__':
    main()
