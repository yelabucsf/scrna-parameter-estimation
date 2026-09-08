"""Figure 6C and 6D - precomputed mode against the default, for DM and DV.

Port of publication/cellxgene/cxg_comparison/cellxgene_comparison.ipynb cells 15-39.

Two routes to the same comparison, classical monocytes against CD4 T cells in one donor
of the SLE dataset:

  default      pull the cells from the CELLxGENE census and run memento end to end
  precomputed  read that donor's estimators straight out of the TileDB cube and compute
               p-values from the stored means, variances and standard errors

If the precomputed path is sound, the two agree.

Census note: the notebook used release 2023-10-30, which CZI has retired. `config` points
at 2023-12-15, the nearest surviving release. The Lupus dataset carries the same
1,263,676 cells in every available release, so the comparison is unaffected.

Cube note: the stored cube's dimension names are rotated relative to their contents -- the
column called `feature_id` holds cell types, `cell_type` holds dataset ids and
`dataset_id` holds features. The notebook corrected this on read and so does this script.
"""

import os

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import tiledb
import tiledbsoma as soma
from somacore import AxisQuery

import config

config.add_repo_to_path()
import memento  # noqa: E402

CENSUS_URI = ('s3://cellxgene-data-public/cell-census/'
              f'{config.CENSUS_VERSION}/soma/census_data/homo_sapiens')
BUFFER_BYTES = 2 ** 31
CACHED_MEMENTO = 'panel_cd_memento_default.csv'


def load_donor_cells():
    """The one donor of the SLE dataset the comparison runs on."""
    value_filter = (f"is_primary_data == True and dataset_id == '{config.LUPUS_DATASET_ID}' "
                    f"and donor_id == '{config.LUPUS_DONOR}'")
    context = soma.SOMATileDBContext().replace(tiledb_config={
        'soma.init_buffer_bytes': BUFFER_BYTES,
        'vfs.s3.region': 'us-west-2',
        'vfs.s3.no_sign_request': True})

    with soma.Experiment.open(uri=CENSUS_URI, context=context) as experiment:
        query = experiment.axis_query(
            measurement_name='RNA',
            obs_query=AxisQuery(value_filter=value_filter),
            # All genes, so size factors come out right even when testing a subset.
            var_query=AxisQuery())
        adata = query.to_anndata('raw')
    adata.var.index = adata.var['feature_id'].tolist()
    return adata


def run_default(adata, ct1, ct2):
    """Full memento on the raw cells."""
    subset = adata[adata.obs['cell_type'].isin([ct1, ct2])].copy()
    subset.obs['q'] = config.CAPTURE_RATE
    memento.setup_memento(subset, q_column='q', trim_percent=1,
                          filter_mean_thresh=0.1, shrinkage=0)
    memento.create_groups(subset, label_columns=['cell_type'])
    memento.compute_1d_moments(subset, min_perc_group=0.7)

    groups = memento.get_groups(subset)
    groups['intercept'] = 1
    memento.ht_1d_moments(
        subset, covariate=groups[['intercept']],
        treatment=(groups[['cell_type']] == ct2).astype(float),
        num_boot=5000, verbose=1, num_cpus=8, resample_rep=False, approx=True)
    return memento.get_1d_ht_result(subset)


def _fit_mv_regressor(mean, var):
    usable = (mean > 0) & (var > 0)
    return np.polyfit(np.log(mean[usable]), np.log(var[usable]), 2)


def add_residual_variance(frame):
    """Variance with the mean-variance trend divided out."""
    mean, var = frame['mean'], frame['var']
    usable = (mean > 0) & (var > 0)
    trend = np.poly1d(_fit_mv_regressor(mean, var))
    residual = np.full(mean.shape, np.nan)
    with np.errstate(invalid='ignore'):
        residual[usable] = np.exp(np.log(var[usable]) - trend(np.log(mean[usable])))
    frame['res_var'] = residual


def load_estimators():
    """This donor's slice of the precomputed cube, with the rotated names corrected."""
    estimators = tiledb.open(config.CUBE_PATH).df[:]
    estimators = estimators.query(f'donor_id == "{config.LUPUS_DONOR}"').rename(columns={
        'feature_id': 'cell_type', 'cell_type': 'dataset_id', 'dataset_id': 'feature_id'})
    return estimators


def run_precomputed(estimators, ct1, ct2):
    """p-values from the stored estimators, with no access to the cells."""
    first = estimators.query('cell_type == @ct1').copy()
    second = estimators.query('cell_type == @ct2').copy()
    add_residual_variance(first)
    add_residual_variance(second)
    merged = first.merge(second, on='feature_id', suffixes=('_ct1', '_ct2'))

    # Delta method on the log scale: the standard error of log(mean) from the SEM.
    lfc = np.log(merged['mean_ct2'].values / merged['mean_ct1'].values)
    log_se = [(np.log(merged[f'mean_{s}'] + merged[f'sem_{s}'])
               - np.log(merged[f'mean_{s}'] - merged[f'sem_{s}'])) / 2
              for s in ('ct1', 'ct2')]
    se_lfc = np.sqrt(log_se[0] ** 2 + log_se[1] ** 2).values
    de_pval = stats.norm.sf(np.abs(lfc), loc=0, scale=se_lfc) * 2

    dv_lfc = np.log(merged['res_var_ct2'].values / merged['res_var_ct1'].values)
    se_dv_lfc = np.sqrt(merged['selv_ct1'] ** 2 + merged['selv_ct2'] ** 2).values
    dv_pval = stats.norm.sf(np.abs(dv_lfc), loc=0, scale=se_dv_lfc) * 2

    return pd.DataFrame({
        'gene': merged['feature_id'].values,
        'cxg_de_coef': lfc, 'cxg_de_pval': de_pval,
        'cxg_dv_coef': dv_lfc, 'cxg_dv_pval': dv_pval}).dropna()


def scatter(ax, x, y, title, limit):
    ax.scatter(x, y, s=3, color=config.MEMENTO_COLOR)
    ax.plot([0, limit], [0, limit], '--', color='k', lw=1)
    ax.set_title(title)
    ax.set_xlabel('Default')
    ax.set_ylabel('Precomputed')
    return stats.pearsonr(x, y)[0]


def main():
    ct1, ct2 = config.COMPARISON_CELL_TYPES
    config.set_style()

    cache = config.intermediate_path(CACHED_MEMENTO)
    if os.path.exists(cache):
        print(f'reusing cached default-mode result: {cache}')
        default = pd.read_csv(cache)
    else:
        adata = load_donor_cells()
        print(f'{adata.shape[0]} cells x {adata.shape[1]} genes for donor '
              f'{config.LUPUS_DONOR}', flush=True)
        default = run_default(adata, ct1, ct2)
        default.to_csv(cache, index=False)

    precomputed = run_precomputed(load_estimators(), ct1, ct2)
    merged = default.merge(precomputed, on='gene')
    for prefix, column in [('de', 'de_pval'), ('dv', 'dv_pval')]:
        merged[f'mem_{prefix}_logp'] = -np.log10(merged[column])
        merged[f'cxg_{prefix}_logp'] = -np.log10(merged[f'cxg_{prefix}_pval'])
    print(f'{merged.shape[0]} genes tested by both routes')

    fig, axes = plt.subplots(1, 2, figsize=(6, 2.4))
    plt.subplots_adjust(wspace=0.5)

    dm = merged.query('mem_de_logp < 200 & cxg_de_logp < 200')
    r_dm = scatter(axes[0], dm['mem_de_logp'], dm['cxg_de_logp'],
                   'mean -log10(P)', 200)
    dv = merged.query('dv_coef < 6 & cxg_dv_coef < 6 & mem_dv_logp < 25 & cxg_dv_logp < 25')
    r_dv = scatter(axes[1], dv['mem_dv_logp'], dv['cxg_dv_logp'],
                   'variability -log10(P)', 20)
    print(f'panel C: {dm.shape[0]} genes, Pearson r = {r_dm:.3f}')
    print(f'panel D: {dv.shape[0]} genes, Pearson r = {r_dv:.3f}')

    fig.savefig(config.figure_path('figure6CD.pdf'), bbox_inches='tight')
    fig.savefig(config.figure_path('figure6CD.png'), bbox_inches='tight', dpi=300)
    merged.to_csv(config.intermediate_path('panel_cd_comparison.csv'), index=False)
    print('wrote figure6CD')


if __name__ == '__main__':
    main()
