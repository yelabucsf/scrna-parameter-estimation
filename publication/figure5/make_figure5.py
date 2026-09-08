"""Regenerate every panel of Figure 5 and assemble them into one sheet.

Assumes the data tree has been built (see README.md):
    python data_manifest.py link
"""

import argparse

import matplotlib
matplotlib.use('Agg')

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

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
            module.main()

    fig, axes = plt.subplots(
        len(SHEET), 1, figsize=(11, 18),
        gridspec_kw={'height_ratios': [HEIGHTS[stem] for stem, _ in SHEET]})
    for ax, (stem, label) in zip(axes, SHEET):
        ax.imshow(mpimg.imread(config.figure_path(f'{stem}.png')))
        ax.set_axis_off()
        ax.set_title(label, loc='left', fontweight='bold')
    fig.savefig(config.figure_path('figure5.png'), bbox_inches='tight', dpi=150)
    print('wrote figure5.png')


if __name__ == '__main__':
    main()
