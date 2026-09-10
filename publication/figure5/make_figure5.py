"""Regenerate every panel of Figure 5 and assemble them into one sheet.

Assumes the data bundle has been downloaded and MEMENTO_DATA_PATH points at it; see
README.md.

Panels F-I need the controlled-access genotypes (dbGaP phs002812.v1.p1) and are skipped
if those are absent, which is the normal case when working from the public bundle.
"""

import argparse
import os

import matplotlib
matplotlib.use('Agg')

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

import config
import panel_a_qqplots
import panel_b_roc
import panel_c_power
import panel_de_atac
import panel_fi_examples

SHEET = [('figure5A', 'A'), ('figure5B', 'B'), ('figure5C', 'C'),
         ('figure5D', 'D'), ('figure5E', 'E'), ('figure5FI', 'F / G / H / I')]
HEIGHTS = {'figure5A': 1.0, 'figure5B': 0.9, 'figure5C': 0.8,
           'figure5D': 0.8, 'figure5E': 0.9, 'figure5FI': 0.9}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--assemble-only', action='store_true',
                        help='rebuild the sheet from existing panel pngs; panel A is slow')
    args = parser.parse_args()

    if not args.assemble_only:
        for module in [panel_a_qqplots, panel_b_roc, panel_c_power,
                       panel_de_atac, panel_fi_examples]:
            print(f'--- {module.__name__}', flush=True)
            try:
                module.main()
            except SystemExit as reason:
                # Panels F-I need the controlled-access genotypes, which are not in the
                # public bundle. Say so once and carry on: A-E are unaffected, and a
                # partial figure is more useful than none.
                if module is not panel_fi_examples:
                    raise
                print(f'skipping panels F-I:\n{reason}', flush=True)

    panels = [entry for entry in SHEET
              if os.path.exists(config.figure_path(f'{entry[0]}.png'))]
    missing = [label for stem, label in SHEET if (stem, label) not in panels]
    if missing:
        print(f'assembling without: {", ".join(missing)}')

    fig, axes = plt.subplots(
        len(panels), 1, figsize=(11, 18 * len(panels) / len(SHEET)),
        gridspec_kw={'height_ratios': [HEIGHTS[stem] for stem, _ in panels]})
    for ax, (stem, label) in zip(np.atleast_1d(axes), panels):
        ax.imshow(mpimg.imread(config.figure_path(f'{stem}.png')))
        ax.set_axis_off()
        ax.set_title(label, loc='left', fontweight='bold')
    fig.savefig(config.figure_path('figure5.png'), bbox_inches='tight', dpi=150)
    print('wrote figure5.png')


if __name__ == '__main__':
    main()
