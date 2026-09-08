"""Figure 6E - query-time runtime, precomputed against default mode.

Port of publication/cellxgene/cxg_comparison/cellxgene_comparison.ipynb cells 44-53.

These are wall-clock measurements, not derived data: the notebook timed both modes over
an increasing number of pairwise cell-type comparisons in the SLE dataset and recorded
the results inline. Re-measuring the default mode means rerunning memento once per
comparison, which is the ~5 minutes per point the panel is about.

The companion bar chart contrasts one-off precomputation cost: the precomputed mode pays
9 minutes up front so that each later query is nearly free.
"""

import itertools

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np

import config

# Seconds to run every pairwise comparison among the first N cell types, N = 2..10.
PRECOMPUTED_TIMES = np.array([
    0.0445709228515625, 0.12334084510803223, 0.24467754364013672, 0.4052596092224121,
    0.5705969333648682, 0.78609299659729, 1.0434541702270508, 1.396212100982666,
    1.4767653942108154])
DEFAULT_TIMES = np.array([
    16.082085132598877, 46.88715410232544, 86.9343409538269, 131.8815257549286,
    161.52750897407532, 211.55276083946228, 266.297087430954, 295.0561339855194,
    314.1077857017517])

PRECOMPUTE_SECONDS = {'Default': 0.21629118919372559, 'Precomputed': 557.3624973297119}


def comparison_counts(num_points):
    """Pairwise comparisons among the first N cell types, for N = 2, 3, ..."""
    return [len(list(itertools.combinations(range(n), 2)))
            for n in range(2, num_points + 2)]


def main():
    config.set_style()
    counts = comparison_counts(DEFAULT_TIMES.shape[0])

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.2),
                             gridspec_kw={'width_ratios': [2, 1]})
    plt.subplots_adjust(wspace=0.5)

    axes[0].plot(counts, DEFAULT_TIMES / 60, '-o', markeredgecolor='k',
                 color=config.BASELINE_COLOR, label='Default')
    axes[0].plot(counts, PRECOMPUTED_TIMES / 60, '-o', markeredgecolor='k',
                 color=config.MEMENTO_COLOR, label='Precomputed')
    axes[0].set_xlabel('Number of binary comparisons')
    axes[0].set_ylabel('Minutes')
    axes[0].set_title('Query runtime')
    axes[0].legend(frameon=False)

    labels = list(PRECOMPUTE_SECONDS)
    axes[1].bar(range(len(labels)), [PRECOMPUTE_SECONDS[k] for k in labels],
                color=[config.BASELINE_COLOR, config.MEMENTO_COLOR],
                linewidth=2.5, edgecolor='k')
    axes[1].set_xticks(range(len(labels)), labels, rotation=45)
    axes[1].set_ylabel('Runtime (s)')
    axes[1].set_title('Precomputation')

    fig.savefig(config.figure_path('figure6E.pdf'), bbox_inches='tight')
    fig.savefig(config.figure_path('figure6E.png'), bbox_inches='tight', dpi=300)

    speedup = DEFAULT_TIMES / PRECOMPUTED_TIMES
    print(f'{len(counts)} measured points, {counts[0]}-{counts[-1]} comparisons')
    print(f'query speedup {speedup.min():.0f}x to {speedup.max():.0f}x '
          f'(median {np.median(speedup):.0f}x)')
    print(f'precomputation costs {PRECOMPUTE_SECONDS["Precomputed"] / 60:.1f} min up front')
    print('wrote figure6E')


if __name__ == '__main__':
    main()
