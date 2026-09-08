"""Figure 4H - the LGALS3BP locus, with IRF1 and PRDM1 binding sites.

The worked example behind panel 4F: memento's DM and DC analyses predict that IRF1 and
PRDM1 interact at LGALS3BP, and ENCODE ChIP-seq shows both factors binding near its
transcription start site.

The notebook's `encode.Encode` helper streamed peak files from ENCODE at run time and
deleted them afterwards, so nothing was left on the volume. The two peak sets its own
selection rule picks -- highest-ranked IDR thresholded peaks on GRCh38, no audit errors
-- are ENCFF557FUM (IRF1, K562) and ENCFF719BHI (PRDM1, A549); both are staged under
`tfko140/encode_peaks/`. Gene coordinates come from GRCh38Genes.bed, and the TSS is
defined as in `Encode.get_tss_window`: txStart on the + strand, txEnd on the -.
"""

import gzip

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import pandas as pd

import config

GENE = 'LGALS3BP'
PEAK_FILES = {
    'IRF1': config.FIGURE4_DATA + 'panelGH_chipseq/peaks/ENCFF557FUM.bed.gz',
    'PRDM1': config.FIGURE4_DATA + 'panelGH_chipseq/peaks/ENCFF719BHI.bed.gz',
}
PEAK_COLORS = {'IRF1': 'tab:blue', 'PRDM1': 'tab:red'}
BED_COLUMNS = ['chrom', 'txStart', 'txEnd', 'cdsStart', 'cdsEnd', 'symbol', 'name', 'strand']
FLANK = 20000


def gene_locus(symbol):
    """Widest transcript for `symbol`, plus its TSS, from GRCh38Genes.bed."""
    genes = pd.read_csv(config.GENE_BED, sep='\t', names=BED_COLUMNS)
    matches = genes[genes['symbol'] == symbol]
    if matches.empty:
        raise ValueError(f'{symbol} is not in {config.GENE_BED}')
    locus = matches.loc[(matches['txEnd'] - matches['txStart']).idxmax()]
    # Matches Encode.get_tss_window: start of the transcript on +, end on -.
    tss = locus['txStart'] if locus['strand'] == '+' else locus['txEnd']
    return locus, tss


def peaks_near(path, chrom, start, end):
    with gzip.open(path, 'rt') as handle:
        peaks = pd.read_csv(handle, sep='\t', header=None,
                            usecols=[0, 1, 2, 6], names=['chrom', 'start', 'end', 'signal'])
    return peaks.query('chrom == @chrom & end > @start & start < @end')


def main():
    config.set_style()
    locus, tss = gene_locus(GENE)
    chrom = locus['chrom']
    window_start, window_end = locus['txStart'] - FLANK, locus['txEnd'] + FLANK
    print(f'{GENE}: {chrom}:{locus["txStart"]}-{locus["txEnd"]} '
          f'({locus["strand"]} strand), TSS at {tss}')

    fig, ax = plt.subplots(figsize=(7, 2.2))

    # Gene body, drawn as a bar with the TSS marked.
    ax.plot([locus['txStart'], locus['txEnd']], [0, 0], lw=6, color='black',
            solid_capstyle='butt')
    ax.axvline(tss, color='grey', linestyle='--', lw=1)
    ax.annotate(f'{GENE} TSS', xy=(tss, 0.45), ha='center', fontsize=8)

    for offset, (factor, path) in enumerate(PEAK_FILES.items(), start=1):
        found = peaks_near(path, chrom, window_start, window_end)
        print(f'  {factor}: {found.shape[0]} peaks within {FLANK} bp of the gene')
        for _, peak in found.iterrows():
            ax.plot([peak['start'], peak['end']], [offset, offset], lw=8,
                    color=PEAK_COLORS[factor], solid_capstyle='butt')
        ax.annotate(factor, xy=(window_start, offset), va='center', ha='right',
                    fontsize=9, color=PEAK_COLORS[factor])

    ax.set_xlim(window_start, window_end)
    ax.set_ylim(-0.6, len(PEAK_FILES) + 0.8)
    ax.set_yticks([])
    ax.set_xlabel(f'{chrom} position (bp)')
    for spine in ['top', 'right', 'left']:
        ax.spines[spine].set_visible(False)

    fig.savefig(config.figure_path('figure4H.pdf'), bbox_inches='tight')
    fig.savefig(config.figure_path('figure4H.png'), bbox_inches='tight', dpi=300)
    print('wrote figure4H')


if __name__ == '__main__':
    main()
