"""Declarative inventory of the data files Figure 4 depends on.

Same contract as publication/figure{2,3}/data_manifest.py:

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
    add('EF', REQUIRED, 'panelEF_network/cytoscape_SIF.csv', 'tfko140/cytoscape_SIF.csv')
    add('EF', REQUIRED, 'panelEF_network/cytoscape_SIF_explicit.csv',
        'tfko140/cytoscape_SIF_explicit.csv')
    add('EF', REQUIRED, 'panelEF_network/Supplementary_Table_3_Perturb-seq_DM.csv',
        'tables/Supplementary_Table_3_Perturb-seq_DM.csv')
    add('EF', REQUIRED, 'panelEF_network/Supplementary_Table_4_Perturb-seq_DC.csv',
        'tables/Supplementary_Table_4_Perturb-seq_DC.csv')

    # --- Panels G, H: ChIP-seq binding relative to the TSS --------------------
    add('GH', REQUIRED, 'panelGH_chipseq/encode_result.csv', 'tfko140/encode_result.csv')
    add('GH', REQUIRED, 'panelGH_chipseq/GRCh38Genes.bed', 'GRCh38Genes.bed',
        root=config.MISCSEQ_PATH + '/')
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
    build(args.root or config.FIGURE4_DATA, copy=args.command == 'bundle')


if __name__ == '__main__':
    main()
