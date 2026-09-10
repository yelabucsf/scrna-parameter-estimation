"""Figure 2E - runtime vs number of cells.

These are wall-clock measurements, not derived data: the numbers below are the ones
recorded in publication/original/validation/inference/runtime/plots.ipynb, per gene and
normalised by the number of parallel workers each tool was given. BASiCS and scHOT
are R packages, so re-measuring the full panel needs an R installation plus the
simulated datasets that runtime/simulate.ipynb produces; neither is on this machine.
"""

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np

import config

# BASiCS and scHOT were run across 40 and 3 workers respectively.
BASICS_NUM_CELLS = np.array([500, 1000, 5000, 10000, 20000, 50000, 100000])
BASICS_RUNTIME = np.array([77, 150, 1400, 2430, 4020, 8715, 17000]) / 40

SCHOT_NUM_CELLS = np.array([500, 1000, 2000, 10000, 50000, 100000])
SCHOT_RUNTIME = np.array([5.545, 6.392, 8.199, 29.747, 216.559, 273.492]) / 3

MEMENTO_NUM_CELLS = np.array([250, 500, 2500, 5000, 10000, 50000, 100000, 500000]) * 2
MEMENTO_RUNTIME = np.array([
    0.014873566601481314,
    0.01777667029552868,
    0.021939572579455825,
    0.022748108507940443,
    0.027120769728360652,
    0.03643204177761024,
    0.04465434376358209,
    0.11130434910770462,
])

# The published panel stops at 100k cells; the last two memento points go further.
PLOT_LIMIT = -2


def main():
    config.set_style()

    fig, ax = plt.subplots(figsize=(2.4, 2.2))
    ax.plot(BASICS_NUM_CELLS, BASICS_RUNTIME, 'o-', markersize=3, label='BASiCS')
    ax.plot(SCHOT_NUM_CELLS, SCHOT_RUNTIME, 'o-', markersize=3, label='scHOT')
    ax.plot(MEMENTO_NUM_CELLS[:PLOT_LIMIT], MEMENTO_RUNTIME[:PLOT_LIMIT], 'o-',
            markersize=3, color=config.MEMENTO_COLOR, label='memento')
    ax.set_xlabel('number of cells')
    ax.set_ylabel('runtime (s)')
    ax.legend(frameon=False)

    fig.savefig(config.figure_path('figure2E.pdf'), bbox_inches='tight')
    fig.savefig(config.figure_path('figure2E.png'), bbox_inches='tight', dpi=300)
    print('wrote figure2E')


if __name__ == '__main__':
    main()
