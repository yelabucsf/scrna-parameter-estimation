"""Figure 6G - pDCs against cDCs, pooling datasets versus one dataset at a time.

Port of publication/original/cellxgene/rare_celltype_comparison.py, plotted as in
cxg_comparison/cellxgene_crossdata.ipynb.

The point of the panel: plasmacytoid and conventional dendritic cells are rare, so any
single dataset has too few of them to say much. Meta-analysing the precomputed estimators
across many datasets recovers signal that no individual dataset shows.

Each gene is fit by weighted least squares across (cell type, donor) groups, with donor as
a covariate and the estimator standard errors as weights. Run once over all datasets
pooled, then once per dataset.

Cube note: the original script read a purpose-built `estimators_cube_dcs_many`, which is
not on the volume. The full census cube is used instead -- it covers every dendritic cell
type across all 23 datasets, so the same groups are available.
"""

import re
import os

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
import tiledb
import tiledbsoma as soma
from somacore import AxisQuery
from sklearn.linear_model import LinearRegression

import config

SCRIPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      'original', 'cellxgene', 'rare_celltype_comparison.py')
CENSUS_URI = ('s3://cellxgene-data-public/cell-census/'
              f'{config.CENSUS_VERSION}/soma/census_data/homo_sapiens')

DC_CELL_TYPES = ['conventional dendritic cell', 'plasmacytoid dendritic cell',
                 'dendritic cell', 'dendritic cell, human', 'myeloid dendritic cell']
MIN_CELLS_PER_GROUP = 20
MAX_MISSING_FRACTION = 0.7


def dataset_ids():
    """The 23 datasets the original script listed, read from it so the two cannot drift."""
    source = open(SCRIPT).read()
    block = source[source.index('DATASETS = ['):source.index(']', source.index('DATASETS = ['))]
    return [line.strip().strip("',") for line in block.split('\n') if line.strip().startswith("'")]


def treatment_of(cell_type):
    if 'plasma' in cell_type:
        return 'pdc'
    if 'conven' in cell_type or 'myeloid' in cell_type:
        return 'cdc'
    return 'unknown'


def load_cube(datasets):
    with tiledb.open(config.CUBE_PATH) as cube:
        frames = []
        for cell_type in DC_CELL_TYPES:
            try:
                frame = cube.df[cell_type, datasets, :]
            except tiledb.TileDBError:
                continue
            if frame.shape[0]:
                frames.append(frame)
    estimators = pd.concat(frames, ignore_index=True)
    estimators['treatment'] = estimators['cell_type'].apply(treatment_of)
    return estimators.query('treatment != "unknown"').copy()


def cell_counts(datasets):
    """Cells per (dataset, cell type, donor), from census metadata only."""
    dataset_filter = ' or '.join(f"dataset_id == '{d}'" for d in datasets)
    type_filter = ' or '.join(f"cell_type == '{c}'" for c in DC_CELL_TYPES)
    context = soma.SOMATileDBContext().replace(tiledb_config={
        'vfs.s3.region': 'us-west-2', 'vfs.s3.no_sign_request': True})
    with soma.Experiment.open(uri=CENSUS_URI, context=context) as experiment:
        query = experiment.axis_query(
            measurement_name='RNA',
            obs_query=AxisQuery(value_filter=f'({dataset_filter}) and ({type_filter})'))
        obs = query.obs(column_names=['dataset_id', 'cell_type', 'donor_id']).concat().to_pandas()
    return obs.groupby(['dataset_id', 'cell_type', 'donor_id']).size().rename('n_cells')


def design_matrix(design):
    """Treatment, mean-centred donor dummies, and their interactions."""
    covariates = design.iloc[:, 1:].astype(float)
    covariates -= covariates.mean(axis=0)
    treatment = design.iloc[:, [0]].astype(float)
    interactions = covariates.multiply(treatment.iloc[:, 0], axis=0)
    interactions.columns = [f'interaction_{c}' for c in covariates.columns]
    covariates = sm.add_constant(pd.concat([covariates, interactions], axis=1))
    return pd.concat([treatment, covariates], axis=1).values.astype(float)


def weighted_least_squares(X, y, variances):
    model = LinearRegression(fit_intercept=False).fit(X, y)
    coef = model.coef_[0]
    try:
        beta_var = np.diag(np.linalg.pinv((X.T * (1 / variances)) @ X))
    except np.linalg.LinAlgError:
        return np.nan, np.nan, np.nan
    se = np.sqrt(beta_var[0])
    z = np.abs(coef) / se
    return coef, se, stats.norm.sf(z) * 2


def compare(estimators, counts):
    """One WLS fit per gene across the (cell type, donor) groups available."""
    estimators = estimators.copy()
    estimators['group_name'] = estimators['treatment'] + '_' + estimators['donor_id']
    # A donor only informs the contrast if it contributes both cell types.
    usable = (estimators[['treatment', 'donor_id']].drop_duplicates()
              .groupby('donor_id').size())
    estimators = estimators[estimators['donor_id'].isin(usable[usable > 1].index)]
    if estimators.empty:
        return pd.DataFrame(columns=['feature_id', 'coef', 'se', 'pval'])
    # One (donor, gene) can appear several times -- the same donor's dendritic cells
    # measured under different assays or suspension types, across five dendritic-cell
    # labels. Here that is 102,260 of 831,157 rows. `drop_duplicates` keeps whichever
    # comes first, and TileDB guarantees no row order, so without an explicit sort the
    # panel's output depends on how the cube happens to be laid out on disk: the same
    # data in a differently-fragmented array shifts the count of significant genes by
    # about 17. Sort first so the choice is a property of the data, not of the storage.
    estimators = estimators.sort_values(
        ['group_name', 'feature_id', 'dataset_id', 'cell_type', 'assay',
         'suspension_type']).drop_duplicates(subset=['group_name', 'feature_id'])

    mean = estimators.pivot(index='group_name', columns='feature_id', values='mean')
    sem = estimators.pivot(index='group_name', columns='feature_id', values='sem')
    groups = estimators.drop_duplicates('group_name').set_index('group_name')

    group_cells = groups.set_index(['dataset_id', 'cell_type', 'donor_id']).index.map(
        counts).to_numpy(dtype=float)

    keep = mean.columns[mean.isnull().values.mean(axis=0) < MAX_MISSING_FRACTION]
    mean, sem = mean.loc[groups.index, keep], sem.loc[groups.index, keep]

    design = pd.DataFrame({
        'treatment': (groups['treatment'] == 'pdc').astype(float).values,
        'donor_id': groups['donor_id'].values}, index=groups.index)

    rows = []
    for feature in keep:
        m, s = mean[feature].values, sem[feature].values
        # Work in log space; the delta method turns the SEM into a log-scale SE.
        log_mean = np.log(m)
        log_se = (np.log(m + s) - np.log(m - s)) / 2
        usable_rows = np.isfinite(log_mean) & np.isfinite(log_se) & (group_cells > MIN_CELLS_PER_GROUP)
        if usable_rows.sum() < 2:
            continue
        sample = design.iloc[usable_rows]
        repeated = sample.groupby('donor_id').size()
        final = sample['donor_id'].isin(repeated[repeated > 1].index).values
        if final.sum() < 2:
            continue
        dummies = pd.get_dummies(sample.iloc[final], columns=['donor_id'], drop_first=True)
        coef, se, pval = weighted_least_squares(
            design_matrix(dummies), log_mean[usable_rows][final],
            log_se[usable_rows][final] ** 2)
        rows.append((feature, coef, se, pval))
    return pd.DataFrame(rows, columns=['feature_id', 'coef', 'se', 'pval'])


def draw_qq(ax, frame, color, size, label):
    frame = frame[(frame['pval'] != 1.0) & frame['pval'].notna()]
    pvalues = np.sort(frame['pval'].values)
    pvalues = pvalues[np.isfinite(pvalues) & (pvalues > 0)]
    if pvalues.shape[0] < 2:
        return 0
    expected = np.linspace(1e-6, 1, pvalues.shape[0])
    ax.scatter(-np.log10(expected), -np.log10(pvalues), s=size, color=color, label=label)
    return pvalues.shape[0]


def main():
    config.set_style()
    datasets = dataset_ids()
    print(f'{len(datasets)} datasets', flush=True)

    counts = cell_counts(datasets)
    print(f'{counts.shape[0]} (dataset, cell type, donor) groups', flush=True)
    estimators = load_cube(datasets)
    print(f'{estimators.shape[0]} cube rows over '
          f'{estimators.dataset_id.nunique()} datasets', flush=True)

    pooled = compare(estimators, counts)
    pooled.to_csv(config.intermediate_path('rare_ct_whole.csv'), index=False)
    print(f'pooled: {pooled.shape[0]} genes, '
          f'{(pooled["pval"] < 0.05).sum()} at p < 0.05', flush=True)

    per_dataset = {}
    for dataset in datasets:
        result = compare(estimators.query('dataset_id == @dataset'), counts)
        if result.empty:
            continue
        result.to_csv(config.intermediate_path(f'rare_ct_{dataset}.csv'), index=False)
        per_dataset[dataset] = result
    print(f'{len(per_dataset)} datasets yielded a per-dataset fit')

    fig, ax = plt.subplots(figsize=(3, 2.6))
    for dataset, result in per_dataset.items():
        draw_qq(ax, result, 'gray', 0.3, None)
    kept = draw_qq(ax, pooled, config.MEMENTO_COLOR, 5, 'all datasets pooled')
    limit = 8
    ax.plot([0, limit], [0, limit], '--', color='k', lw=1)
    ax.set_xlim(-0.2, 4)
    ax.set_xlabel('Expected')
    ax.set_ylabel('Observed\n-log10(P-value)')
    ax.legend(frameon=False, markerscale=2)

    fig.savefig(config.figure_path('figure6G.pdf'), bbox_inches='tight')
    fig.savefig(config.figure_path('figure6G.png'), bbox_inches='tight', dpi=300)
    print(f'wrote figure6G ({kept} pooled genes plotted)')


if __name__ == '__main__':
    main()
