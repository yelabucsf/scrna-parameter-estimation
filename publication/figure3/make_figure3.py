"""Regenerate every panel of Figure 3 and assemble them into one sheet.

Assumes the data bundle has been downloaded, MEMENTO_DATA_PATH points at it, and the
compute step has run (see README.md):
    python run_isg_clustering.py correlations
"""

import matplotlib
matplotlib.use('Agg')

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

import config
import panel_a_umaps
import panel_bc_mean_response
import panel_d_coexpression_network
import panel_efg_variability

# (file stem, label) in published order; B and C come out of one script, E and F share a figure.
SHEET = [('figure3A', 'A'), ('figure3B', 'B'), ('figure3C', 'C'),
         ('figure3D', 'D'), ('figure3EF', 'E / F'), ('figure3G', 'G')]
HEIGHTS = {'figure3A': 1.0, 'figure3B': 0.8, 'figure3C': 1.6,
           'figure3D': 0.9, 'figure3EF': 0.9, 'figure3G': 0.9}


def main():
    for module in [panel_a_umaps, panel_bc_mean_response,
                   panel_d_coexpression_network, panel_efg_variability]:
        print(f'--- {module.__name__}', flush=True)
        module.main()

    fig, axes = plt.subplots(
        len(SHEET), 1, figsize=(11, 22),
        gridspec_kw={'height_ratios': [HEIGHTS[stem] for stem, _ in SHEET]})
    for ax, (stem, label) in zip(axes, SHEET):
        ax.imshow(mpimg.imread(config.figure_path(f'{stem}.png')))
        ax.set_axis_off()
        ax.set_title(label, loc='left', fontweight='bold')
    fig.savefig(config.figure_path('figure3.png'), bbox_inches='tight', dpi=150)
    print('wrote figure3.png')


if __name__ == '__main__':
    main()
