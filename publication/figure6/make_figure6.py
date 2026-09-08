"""Regenerate the analysis panels of Figure 6 and assemble them into one sheet.

Panels A, B and F are figure art with no generating code, so this covers C, D, E and G.
Assumes the estimators cube has been unpacked (see README.md).
"""

import argparse

import matplotlib
matplotlib.use('Agg')

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

import config
import panel_cd_comparison
import panel_e_runtime
import panel_g_crossdataset

SHEET = [('figure6CD', 'C / D'), ('figure6E', 'E'), ('figure6G', 'G')]
HEIGHTS = {'figure6CD': 1.0, 'figure6E': 0.9, 'figure6G': 1.1}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--assemble-only', action='store_true',
                        help='rebuild the sheet from existing panel pngs')
    args = parser.parse_args()

    if not args.assemble_only:
        for module in [panel_cd_comparison, panel_e_runtime, panel_g_crossdataset]:
            print(f'--- {module.__name__}', flush=True)
            module.main()

    fig, axes = plt.subplots(
        len(SHEET), 1, figsize=(11, 11),
        gridspec_kw={'height_ratios': [HEIGHTS[stem] for stem, _ in SHEET]})
    for ax, (stem, label) in zip(axes, SHEET):
        ax.imshow(mpimg.imread(config.figure_path(f'{stem}.png')))
        ax.set_axis_off()
        ax.set_title(label, loc='left', fontweight='bold')
    fig.savefig(config.figure_path('figure6.png'), bbox_inches='tight', dpi=150)
    print('wrote figure6.png')


if __name__ == '__main__':
    main()
