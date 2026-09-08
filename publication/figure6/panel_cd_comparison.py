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

Cube note: the notebook rotated the cube's dimension names on read, correcting an
earlier build in which they were mislabelled. The cube on the volume
(`estimators_cube_v2`, built 2023-10-23) is labelled correctly -- its `cell_type`
dimension really does hold cell types -- so that rename is *not* applied here; doing so
would scramble the query.
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
CACHED_CELLS = 'panel_cd_donor_cells.h5ad'


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


def load_cells():
    """Donor cells, cached locally so the census is queried only once."""
    cached = config.intermediate_path(CACHED_CELLS)
    if os.path.exists(cached):
        import scanpy as sc
        adata = sc.read_h5ad(cached)
    else:
        adata = load_donor_cells()
        adata.write(cached)
    print(f'{adata.shape[0]} cells x {adata.shape[1]} genes for donor '
          f'{config.LUPUS_DONOR}', flush=True)
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
    # get_groups now returns label columns already numerically encoded, so the notebook's
    # `groups[['cell_type']] == ct2` compares floats against a string and yields an
    # all-zero treatment -- which today's memento then drops as constant, leaving an
    # empty design. The group labels ('sg^<cell type>') still carry the name, so the
    # treatment is derived from those instead.
    labels = groups.index.str.replace(r'^sg\^', '', regex=True)
    treatment = pd.DataFrame({'cell_type': (labels == ct2).astype(float)}, index=groups.index)
    if treatment['cell_type'].nunique() < 2:
        raise ValueError(f'treatment is constant; group labels were {list(labels)}')

    memento.ht_1d_moments(
        subset, covariate=groups[['intercept']], treatment=treatment,
        # `approx` was a boolean in the version the notebook used and now names the null
        # approximation. 'norm' is the default and gives the wider usable range here;
        # 'gdp' saturates around -log10(P) = 6 on this two-group comparison.
        num_boot=5000, verbose=1, num_cpus=8, resample_rep=False, approx='norm')
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


def load_estimators(cell_types):
    """This donor's slice of the precomputed cube.

    Sliced on the cube's dimensions rather than read whole: the array is 17 GB, and the
    comparison needs two cell types in one dataset for one donor.
    """
    with tiledb.open(config.CUBE_PATH) as cube:
        frames = [cube.df[cell_type, config.LUPUS_DATASET_ID, :] for cell_type in cell_types]
    estimators = pd.concat(frames, ignore_index=True)
    estimators = estimators.query(f'donor_id == "{config.LUPUS_DONOR}"')
    # The cube on the volume was built with the variance fields largely unpopulated, which
    # caps how many genes panel D can compare. Report it rather than let it surface as a
    # mysteriously small point cloud.
    for cell_type, group in estimators.groupby('cell_type', observed=True):
        print(f'  {cell_type}: {group.shape[0]} genes, '
              f'{(group["var"] > 0).sum()} with a non-zero variance', flush=True)
    return estimators


def normalize_to_relative_abundance(frame):
    """Put a cell type's stored means on a common scale.

    The cube estimates each (cell type, dataset, donor) group independently, so the
    stored means of two cell types do not sum to the same total -- here 41.42 for
    monocytes against 14.48 for CD4 T cells. Taking a ratio without rescaling shifts
    *every* gene's log fold change by log(14.48/41.42) = -1.05, which correlation against
    the default route cannot see (it is a pure intercept) but which throws the p-values
    off by orders of magnitude. The default route normalizes both cell types together, so
    matching it means dividing each group by its own total first.

    `sem` is rescaled with it, which leaves the delta-method log-scale SE untouched --
    (log(m+s) - log(m-s))/2 is invariant when m and s are scaled together. `var` needs no
    rescaling: a constant factor shifts log(var) by a constant, which the mean-variance
    trend absorbs when it is refit.
    """
    total = frame['mean'].sum()
    frame['mean'] = frame['mean'] / total
    frame['sem'] = frame['sem'] / total


def run_precomputed(estimators, ct1, ct2):
    """p-values from the stored estimators, with no access to the cells."""
    first = estimators.query('cell_type == @ct1').copy()
    second = estimators.query('cell_type == @ct2').copy()
    for frame in (first, second):
        normalize_to_relative_abundance(frame)
        add_residual_variance(frame)
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

    # No global dropna: the mean comparison needs only mean and sem, which are defined
    # for every gene, while the variability comparison needs a residual variance, which
    # is undefined wherever the stored variance collapsed to zero. Dropping rows for both
    # at once would throw away most of panel C for panel D's sake.
    return pd.DataFrame({
        'gene': merged['feature_id'].values,
        'cxg_de_coef': lfc, 'cxg_de_pval': de_pval,
        'cxg_dv_coef': dv_lfc, 'cxg_dv_pval': dv_pval})


def scatter(ax, x, y, title, limit):
    ax.scatter(x, y, s=3, color=config.MEMENTO_COLOR)
    if limit is None:
        low = float(min(x.min(), y.min()))
        high = float(max(x.max(), y.max()))
        ax.plot([low, high], [low, high], '--', color='k', lw=1)
    else:
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
        default = run_default(load_cells(), ct1, ct2)
        default.to_csv(cache, index=False)

    precomputed = run_precomputed(load_estimators([ct1, ct2]), ct1, ct2)
    merged = default.merge(precomputed, on='gene')
    for prefix, column in [('de', 'de_pval'), ('dv', 'dv_pval')]:
        merged[f'mem_{prefix}_logp'] = -np.log10(merged[column])
        merged[f'cxg_{prefix}_logp'] = -np.log10(merged[f'cxg_{prefix}_pval'])
    print(f'{merged.shape[0]} genes tested by both routes')

    fig, axes = plt.subplots(1, 4, figsize=(12, 2.4))
    plt.subplots_adjust(wspace=0.55)

    dm = merged.query('mem_de_logp < 200 & cxg_de_logp < 200').dropna(
        subset=['mem_de_logp', 'cxg_de_logp'])
    dv = merged.query(
        'dv_coef < 6 & cxg_dv_coef < 6 & mem_dv_logp < 25 & cxg_dv_logp < 25').dropna(
        subset=['mem_dv_logp', 'cxg_dv_logp'])

    # The effect sizes are where the two routes should agree outright; the p-values carry
    # a systematic offset because the precomputed route tests analytically from stored
    # standard errors while the default route bootstraps.
    coef_dm = merged.dropna(subset=['de_coef', 'cxg_de_coef'])
    r_coef_dm = scatter(axes[0], coef_dm['de_coef'], coef_dm['cxg_de_coef'],
                        f'mean LFC (n={coef_dm.shape[0]})', None)
    r_dm = scatter(axes[1], dm['mem_de_logp'], dm['cxg_de_logp'],
                   f'mean -log10(P) (n={dm.shape[0]})', 200)
    r_coef_dv = scatter(axes[2], dv['dv_coef'], dv['cxg_dv_coef'],
                        f'variability LFC (n={dv.shape[0]})', None)
    r_dv = scatter(axes[3], dv['mem_dv_logp'], dv['cxg_dv_logp'],
                   f'variability -log10(P) (n={dv.shape[0]})', 20)

    print(f'panel C  mean LFC:        {coef_dm.shape[0]:5} genes, Pearson r = {r_coef_dm:.3f}')
    print(f'panel C  mean -log10(P):  {dm.shape[0]:5} genes, Pearson r = {r_dm:.3f}')
    print(f'panel D  variability LFC: {dv.shape[0]:5} genes, Pearson r = {r_coef_dv:.3f}')
    print(f'panel D  var -log10(P):   {dv.shape[0]:5} genes, Pearson r = {r_dv:.3f}')

    fig.savefig(config.figure_path('figure6CD.pdf'), bbox_inches='tight')
    fig.savefig(config.figure_path('figure6CD.png'), bbox_inches='tight', dpi=300)
    merged.to_csv(config.intermediate_path('panel_cd_comparison.csv'), index=False)
    print('wrote figure6CD')


if __name__ == '__main__':
    main()
