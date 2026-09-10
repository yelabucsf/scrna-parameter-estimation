"""Figure 5A - QQ plots for eQTLs, vQTLs and cQTLs.

Port of publication/original/genetics/run_memento/qqplots.ipynb.

All three panels pool the tests across both ancestry groups and all six cell types, then
keep only variants whose minor allele frequency exceeds 10% in their own population --
rare variants dominate the tail otherwise. The eQTL panel overlays memento against the
pseudobulk Matrix eQTL run on the same data.

The notebook also merged per-gene mean expression into the vQTL table; that column is not
used by the plot, so the twelve single-cell h5ads it required are not loaded here.
"""

import os

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats

import config

MEMENTO_DIR = config.FIGURE5_DATA + 'panelA_qq/memento/'
MATEQTL_DIR = config.FIGURE5_DATA + 'panelA_qq/mateqtl/'
GENOTYPE_DIR = config.FIGURE5_DATA + 'genotypes/'
# memento reports a handful of astronomically small eQTL p-values; the notebook capped
# the plotted range via a `log10p` column that the stored result files do not carry, so
# the same cap is applied directly to the p-value here.
MAX_LOG10P = 50
MIN_PVALUE = 10.0 ** -MAX_LOG10P


GENOTYPE_VALUES = (0, 1, 2)
GENOTYPE_CHUNK = 200_000


MAF_DIR = config.FIGURE5_DATA + 'panelA_qq/maf/'


def minor_allele_frequencies():
    """Frequency of the rarest observed genotype call per variant, per population.

    Reads the precomputed per-variant frequencies if they are present, and falls back to
    deriving them from the genotype matrices.

    The precomputed form is what the published data bundle ships. The genotype matrices
    are individual-level data from dbGaP phs002812 and cannot be redistributed, but this
    panel never needs them: it only wants one aggregate frequency per variant, which
    identifies nobody. `write_minor_allele_frequencies` below regenerates the csvs from
    the matrices for anyone with dbGaP access.
    """
    precomputed = [MAF_DIR + f'{population}_maf.csv' for population in config.POPULATIONS]
    if all(os.path.exists(path) for path in precomputed):
        frames = [pd.read_csv(path) for path in precomputed]
        for frame, population in zip(frames, config.POPULATIONS):
            print(f'  {population}: {frame.shape[0]} variants (precomputed)', flush=True)
        return pd.concat(frames, ignore_index=True)

    if not os.path.isdir(GENOTYPE_DIR):
        raise SystemExit(
            f'no precomputed allele frequencies under {MAF_DIR}, and no genotype\n'
            'matrices to derive them from. The genotypes are controlled-access (dbGaP\n'
            'phs002812.v1.p1) and are not part of the published bundle; the bundle ships\n'
            'the aggregate frequencies this panel actually uses. If you are seeing this,\n'
            'the download is incomplete.')

    return _derive_minor_allele_frequencies()


def _derive_minor_allele_frequencies():
    """Compute the frequencies from the genotype matrices. Needs dbGaP access.

    The notebook did this with a row-wise `value_counts` lambda. Over 3.3M variants that
    is hours of Python-level calls, so it is vectorised here and read in chunks to keep
    the genotype matrices out of memory. Same quantity: for each variant, the smallest
    non-zero share among the genotype classes actually observed.
    """
    frames = []
    for population in config.POPULATIONS:
        minima, variants = [], []
        reader = pd.read_csv(GENOTYPE_DIR + f'{population}_genos.tsv', sep='\t',
                             index_col=0, chunksize=GENOTYPE_CHUNK)
        for chunk in reader:
            values = chunk.to_numpy()
            counts = np.stack([(values == g).sum(axis=1) for g in GENOTYPE_VALUES])
            totals = counts.sum(axis=0)
            shares = np.divide(counts, totals, out=np.zeros_like(counts, dtype=float),
                               where=totals > 0)
            # Ignore classes that never occur, then take the rarest of the rest.
            shares[counts == 0] = np.inf
            minima.append(shares.min(axis=0))
            variants.append(chunk.index.to_numpy())
        frame = pd.DataFrame({'CHROM:POS': np.concatenate(variants),
                              'min_count': np.concatenate(minima)})
        frame['pop'] = population
        frames.append(frame)
        print(f'  {population}: {frame.shape[0]} variants', flush=True)
    return pd.concat(frames, ignore_index=True)


def write_minor_allele_frequencies(out_dir=None):
    """Derive the per-variant frequencies and write them out, one csv per population.

    Maintainer step, run once against the controlled-access genotypes to produce the
    aggregate files the public bundle ships in their place.
    """
    out_dir = out_dir or MAF_DIR
    os.makedirs(out_dir, exist_ok=True)
    table = _derive_minor_allele_frequencies()
    for population, group in table.groupby('pop'):
        path = os.path.join(out_dir, f'{population}_maf.csv')
        group.to_csv(path, index=False)
        print(f'wrote {path} ({group.shape[0]} variants)')


def load_tests(suffix, directory=MEMENTO_DIR):
    """Concatenate one result file per population and cell type."""
    frames = []
    for population in config.POPULATIONS:
        for cell_type in config.CELL_TYPES:
            path = f'{directory}{population}_{cell_type}{suffix}'
            frame = pd.read_csv(path, sep='\t' if path.endswith('_all_hg19.csv') else ',')
            frame['pop'] = population
            frame['ct'] = cell_type
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def apply_maf_filter(tests, frequencies, variant_column):
    merged = tests.merge(frequencies, left_on=[variant_column, 'pop'],
                         right_on=['CHROM:POS', 'pop'], how='left')
    return merged.query('min_count > @config.MIN_ALLELE_FREQUENCY')


def plot_qq(ax, pvalues, color, label):
    pvalues = np.asarray(pvalues)
    pvalues = pvalues[np.isfinite(pvalues) & (pvalues > 0)]
    expected = stats.uniform.ppf(np.linspace(0, 1, pvalues.shape[0]))
    ax.scatter(-np.log10(expected), -np.log10(np.sort(pvalues)), s=1,
               color=color, label=label)
    return pvalues.shape[0]


def main():
    config.set_style()
    frequencies = minor_allele_frequencies()
    print(f'{frequencies.shape[0]} variant-population pairs with genotypes')

    fig, axes = plt.subplots(1, 3, figsize=(9, 2.6))
    plt.subplots_adjust(wspace=0.4)

    # --- eQTLs: memento against the pseudobulk Matrix eQTL run ---
    memento = apply_maf_filter(load_tests('.csv'), frequencies, 'SNP')
    mateqtl = apply_maf_filter(load_tests('_all_hg19.csv', MATEQTL_DIR), frequencies, 'SNP')
    kept = plot_qq(axes[0], memento.query('`p-value` > @MIN_PVALUE')['p-value'],
                   config.MEMENTO_COLOR, 'memento')
    kept_pb = plot_qq(axes[0], mateqtl['p-value'], config.PSEUDOBULK_COLOR, 'Matrix eQTL')
    axes[0].set_title('Mean QTLs (eQTLs)')
    axes[0].legend(frameon=False, markerscale=6)
    print(f'eQTL: {kept} memento tests, {kept_pb} Matrix eQTL tests')

    # --- vQTLs and cQTLs: memento only ---
    variability = apply_maf_filter(load_tests('_variability.csv'), frequencies, 'tx')
    kept_v = plot_qq(axes[1], variability['dv_pval'], config.MEMENTO_COLOR, 'memento')
    axes[1].set_title('Variability QTLs')
    print(f'vQTL: {kept_v} tests, {(variability["dv_fdr"] < 0.1).sum()} at FDR < 0.1')

    coexpression = apply_maf_filter(load_tests('_coexpression.csv'), frequencies, 'tx')
    kept_c = plot_qq(axes[2], coexpression['corr_pval'], config.MEMENTO_COLOR, 'memento')
    axes[2].set_title('Coexpression QTLs')
    print(f'cQTL: {kept_c} tests, {(coexpression["corr_fdr"] < 0.1).sum()} at FDR < 0.1')

    for ax in axes:
        limit = max(ax.get_xlim()[1], 6)
        ax.plot([0, limit], [0, limit], '--', color='black', lw=1)
        ax.set_xlabel('Theoretical -log10(P)')
        ax.set_ylabel('Observed -log10(P)')

    fig.savefig(config.figure_path('figure5A.pdf'), bbox_inches='tight')
    fig.savefig(config.figure_path('figure5A.png'), bbox_inches='tight', dpi=300)
    print('wrote figure5A')


if __name__ == '__main__':
    main()
