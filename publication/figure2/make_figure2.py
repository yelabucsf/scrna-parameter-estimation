"""Regenerate every panel of Figure 2 and assemble them into one sheet.

Assumes the two compute steps have already run (see README.md):
    python panel_a_run_simulations.py all
    python panel_b_run_correlation.py
"""

import matplotlib
matplotlib.use('Agg')

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

import config
import panel_a_plot
import panel_b_smfish
import panel_c_power_fdr
import panel_d_bulk_concordance
import panel_e_runtime

PANELS = ['A', 'B', 'C', 'D', 'E']


def main():
    for module in [panel_a_plot, panel_b_smfish, panel_c_power_fdr,
                   panel_d_bulk_concordance, panel_e_runtime]:
        print(f'--- {module.__name__}', flush=True)
        module.main()

    fig, axes = plt.subplots(len(PANELS), 1, figsize=(9, 13))
    for ax, panel in zip(axes, PANELS):
        ax.imshow(mpimg.imread(config.figure_path(f'figure2{panel}.png')))
        ax.set_axis_off()
        ax.set_title(panel, loc='left', fontweight='bold')
    fig.savefig(config.figure_path('figure2.png'), bbox_inches='tight', dpi=200)
    print('wrote figure2.png')


if __name__ == '__main__':
    main()
