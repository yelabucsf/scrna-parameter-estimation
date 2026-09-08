"""Figure 2A - Lin's concordance of estimates against simulated ground truth.

Reads the npz/csv written by panel_a_run_simulations.py and reproduces the three
panels from the *_comparison.ipynb notebooks under
publication/original/validation/estimation/simulation/.

The published middle panel also carries a BASiCS curve. BASiCS runs in R against
per-replicate h5ad dumps that are not on the data volume, so it is omitted here.
"""

import os

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config

BASICS_PATH = config.FIGURE2_DATA + 'panelA_simulation/basics/'
BASICS_NUM_CELL = 100
BASICS_CAPTURE_EFFICIENCIES = [0.05, 0.1, 0.2, 0.3, 0.5]
BASICS_TRIALS = 20

# num_cell each published panel is drawn at. The caption says 100 cells for the
# correlation panel, but correlation_comparison.ipynb drew it at 500, and at 100 the
# replicate spread swamps the separation between methods.
PANEL_NUM_CELL = {'mean': 10, 'variance': 100, 'correlation': 500}
Q_LIMIT = {'mean': 0.3, 'variance': 0.6, 'correlation': 0.5}
PANEL_METHODS = {
    'mean': [('hypergeometric', 'memento', config.MEMENTO_COLOR, 'o', '-'),
             ('naive', 'naive', config.BASELINE_COLOR, ',', '--')],
    'variance': [('hypergeometric', 'memento', config.MEMENTO_COLOR, 'o', '-'),
                 ('poisson', 'Poisson', config.BASELINE_COLOR, 's', '-'),
                 ('basics', 'BASiCS', config.BASELINE_COLOR, '^', '-'),
                 ('naive', 'naive', config.BASELINE_COLOR, ',', '--')],
    'correlation': [('hypergeometric', 'memento', config.MEMENTO_COLOR, 'o', '-'),
                    ('poisson', 'Poisson', config.BASELINE_COLOR, 's', '-'),
                    ('naive', 'naive', config.BASELINE_COLOR, ',', '--')],
}


def concordance(x, y, mask, log=True):
    """Lin's concordance correlation coefficient over the masked entries."""
    a, b = (np.log(x[mask]), np.log(y[mask])) if log else (x[mask], y[mask])
    cmat = np.cov(a, b)
    return mask.sum(), 2 * cmat[0, 1] / (cmat[0, 0] + cmat[1, 1] + (a.mean() - b.mean()) ** 2)


def load(quantity):
    estimates = np.load(config.intermediate_path(f'panel_a_{quantity}_estimates.npz'))['estimates']
    meta = pd.read_csv(config.intermediate_path(f'panel_a_{quantity}_metadata.csv'))
    return estimates, meta


def concordance_table_mean():
    """The mean ground truth is a single row (x_param), shared across all replicates."""
    estimates, meta = load('mean')
    truth = estimates[0]
    results = meta.iloc[1:].copy()
    scores = []
    for i in range(1, estimates.shape[0]):
        mask = np.isfinite(np.log(truth)) & np.isfinite(np.log(estimates[i]))
        scores.append(concordance(truth, estimates[i], mask)[1])
    results['concordance'] = scores
    return results


def load_basics_variances(num_genes, columns):
    """BASiCS variance estimates, rescaled onto the simulation's units.

    BASiCS works in its own expression scale, so variance_estimation.py rescaled by the
    mean ratio against the ground-truth means; the same correction is applied here.
    Genes BASiCS could not fit (zero in every cell) come back as NaN and drop out of the
    per-replicate mask, exactly as they did in the published panel.
    """
    true_mean = np.load(config.intermediate_path('panel_a_mean_estimates.npz'))['estimates'][0]

    rows = []
    for q in BASICS_CAPTURE_EFFICIENCIES:
        for trial in range(BASICS_TRIALS):
            path = BASICS_PATH + f'{BASICS_NUM_CELL}_{q}_{trial}_parameters.csv'
            if not os.path.exists(path):
                continue
            params = pd.read_csv(path, index_col=0)
            kept = params.index.astype(int).values
            scale_factor = (params['mu'].values / true_mean[kept]).mean()

            values = np.full(num_genes, np.nan)
            values[kept] = params['variance'].values / scale_factor ** 2
            rows.append(dict(zip(columns, values))
                        | {'q': q, 'num_cell': BASICS_NUM_CELL, 'trial': trial + 1,
                           'method': 'basics'})
    return pd.DataFrame(rows)


def concordance_table_grouped(quantity, log):
    """Variance and correlation carry a per-replicate ground_truth row."""
    estimates, meta = load(quantity)
    columns = [f'v{i}' for i in range(estimates.shape[1])]
    frame = pd.concat([meta, pd.DataFrame(estimates, columns=columns)], axis=1)

    if quantity == 'variance':
        basics = load_basics_variances(estimates.shape[1], columns)
        if not basics.empty:
            frame = pd.concat([frame, basics], ignore_index=True)
            print(f'including BASiCS for {basics.shape[0]} replicates')

    group_keys = ['q', 'num_cell', 'trial']
    rows = []
    for name, group in frame.groupby(group_keys):
        values = group[columns].values
        if quantity == 'correlation':
            num_pairs = int(group['num_pairs'].iloc[0])
            values = values[:, :num_pairs]
            mask = np.all(np.isfinite(values), axis=0)
        else:
            mask = np.all(values > 0, axis=0)
        if mask.sum() < 2:
            continue
        truth = values[0]
        for idx, method in enumerate(group['method']):
            num_used, score = concordance(values[idx], truth, mask, log=log)
            rows.append((*name, method, num_used, score))
    return pd.DataFrame(rows, columns=group_keys + ['method', 'num_valid', 'concordance'])


def plot_curve(data, ax, color, marker, linestyle, label):
    agg = data.groupby('q')['concordance'].agg(['mean', 'std'])
    err = agg['std'] * 3
    ax.plot(agg.index, agg['mean'], marker=marker, color=color, markersize=5,
            linestyle=linestyle, label=label)
    ax.fill_between(agg.index, agg['mean'] - err, agg['mean'] + err, alpha=0.4, color=color)


def make_panel(quantity, results, ax):
    num_cell = PANEL_NUM_CELL[quantity]
    q_limit = Q_LIMIT[quantity]
    subset = results.query('num_cell == @num_cell and q < @q_limit')
    for method, label, color, marker, linestyle in PANEL_METHODS[quantity]:
        rows = subset.query('method == @method')
        if rows.empty:
            continue
        plot_curve(rows, ax, color, marker, linestyle, label)
    ax.set_title(f'{quantity} ({num_cell} cells)')
    ax.set_xlabel('capture efficiency')
    ax.set_ylabel("Lin's concordance")
    ax.legend(frameon=False)


def main():
    config.set_style()

    tables = {
        'mean': concordance_table_mean(),
        'variance': concordance_table_grouped('variance', log=True),
        'correlation': concordance_table_grouped('correlation', log=False),
    }

    fig, axes = plt.subplots(1, 3, figsize=(8, 2.2))
    plt.subplots_adjust(wspace=0.45)
    for ax, quantity in zip(axes, ['mean', 'variance', 'correlation']):
        make_panel(quantity, tables[quantity], ax)
    axes[1].set_ylim(0.49, 1.01)
    fig.savefig(config.figure_path('figure2A.pdf'), bbox_inches='tight')
    fig.savefig(config.figure_path('figure2A.png'), bbox_inches='tight', dpi=300)

    summary = pd.concat([t.assign(quantity=q) for q, t in tables.items()])
    summary.to_csv(config.intermediate_path('panel_a_concordance.csv'), index=False)
    print(summary.groupby(['quantity', 'num_cell', 'method'])['concordance'].mean().round(3))


if __name__ == '__main__':
    main()
