"""Regenerate every panel of Figure 4 and assemble them into one sheet.

Assumes the data bundle has been downloaded, MEMENTO_DATA_PATH points at it, and the
compute step has run (see README.md):
    python run_wt_coexpression.py
"""

import matplotlib
matplotlib.use('Agg')

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

import config
import panel_a_selection
import panel_bc_effect_heatmaps
import panel_d_regulator_target_corr
import panel_ef_networks
import panel_g_chipseq
import panel_h_locus

SHEET = [('figure4A', 'A'), ('figure4B', 'B'), ('figure4C', 'C'), ('figure4D', 'D'),
         ('figure4E', 'E'), ('figure4F', 'F'), ('figure4G', 'G'), ('figure4H', 'H')]
HEIGHTS = {'figure4A': 1.3, 'figure4B': 0.8, 'figure4C': 1.2, 'figure4D': 0.7,
           'figure4E': 1.1, 'figure4F': 1.1, 'figure4G': 0.8, 'figure4H': 0.6}


def main():
    for module in [panel_a_selection, panel_bc_effect_heatmaps,
                   panel_d_regulator_target_corr, panel_ef_networks,
                   panel_g_chipseq, panel_h_locus]:
        print(f'--- {module.__name__}', flush=True)
        module.main()

    fig, axes = plt.subplots(
        len(SHEET), 1, figsize=(11, 26),
        gridspec_kw={'height_ratios': [HEIGHTS[stem] for stem, _ in SHEET]})
    for ax, (stem, label) in zip(axes, SHEET):
        ax.imshow(mpimg.imread(config.figure_path(f'{stem}.png')))
        ax.set_axis_off()
        ax.set_title(label, loc='left', fontweight='bold')
    fig.savefig(config.figure_path('figure4.png'), bbox_inches='tight', dpi=150)
    print('wrote figure4.png')


if __name__ == '__main__':
    main()
