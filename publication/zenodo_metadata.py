"""Generate the Zenodo record metadata for a figure's data bundle.

    python zenodo_metadata.py 3 > metadata_figure3.json

One record per figure, so each gets its own DOI to cite in its README and can be
re-versioned without touching the others. See MAINTAINING.md for the upload runbook.
"""

import argparse
import json
import subprocess
import sys

REPO = 'https://github.com/yelabucsf/scrna-parameter-estimation'
# TODO: confirm before uploading. Taken from the article page rather than from Crossref,
# and a wrong identifier here would be baked into an immutable Zenodo record.
PAPER_DOI = '10.1016/j.cell.2024.09.045'

FIGURES = {
    2: ('method validation and comparisons',
        'Simulated and real-data benchmarks of memento against naive, pseudobulk, '
        'Poisson and BASiCS estimators, smFISH-matched correlation estimates, and '
        'concordance with bulk RNA-seq.'),
    3: ('interferon stimulation in human airway epithelium',
        'Differential mean and variability across four interferon stimulations and five '
        'time points, and the canonical / non-canonical interferon-stimulated gene '
        'analysis.'),
    4: ('Perturb-seq of transcription factor knockouts',
        'Differential expression for 84 sgRNAs against wild type, the coexpression '
        'network, and ChIP-seq binding around perturbed loci.'),
    5: ('eQTL, vQTL and coexpression QTL analysis',
        'QTL summary statistics across two ancestry groups and six cell types, '
        'replication against OneK1K, subsampled-cohort power curves, and enrichment in '
        'cell-type-specific ATAC peaks.'),
    6: ('memento in CZI CELLxGENE Discover',
        'The dendritic-cell slice of the precomputed estimators cube, supporting the '
        'cross-dataset comparison of plasmacytoid against conventional dendritic cells.'),
}

# What a downloader has to know before they start, per figure.
CAVEATS = {
    3: ('Panel F additionally needs supplementary Table S1E of Mostafavi et al., Cell '
        '2016 (mmc2.xls), which is a publisher supplementary file and is not '
        'redistributed here. The figure README gives the DOI to fetch it from.'),
    5: ('Panels F-I additionally need individual-level genotypes, which are '
        'controlled-access under dbGaP phs002812.v1.p1 and are not included. Panels A-E '
        'are complete: the minor-allele-frequency filter panel A applies ships as an '
        'aggregate per-variant frequency, which contains no individual-level data.'),
    6: ('Panels C and D additionally query the public CELLxGENE census at run time, so '
        'this figure needs network access.'),
}


def git_commit():
    try:
        return subprocess.run(['git', 'rev-parse', '--short', 'HEAD'],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return 'unknown'


def metadata(figure):
    subject, description = FIGURES[figure]
    commit = git_commit()
    body = [
        f'<p>Input data to reproduce <strong>Figure {figure}</strong> ({subject}) of '
        f'Kim et al., <em>Cell</em> 2024.</p>',
        f'<p>{description}</p>',
        '<p>Unpack the archive and point <code>MEMENTO_DATA_PATH</code> at the directory '
        'containing it, then run the figure script. Full instructions are in '
        f'<a href="{REPO}/tree/main/publication/figure{figure}">'
        f'publication/figure{figure}/README.md</a>.</p>',
        '<pre>tar -xzf figure%d_data.tar.gz\n'
        'export MEMENTO_DATA_PATH=$PWD\n'
        'python make_figure%d.py</pre>' % (figure, figure),
        '<p>Verify the download against the accompanying <code>.sha256</code> file '
        'before unpacking.</p>',
    ]
    if figure in CAVEATS:
        body.append(f'<p><strong>Note.</strong> {CAVEATS[figure]}</p>')

    return {
        'metadata': {
            'title': f'memento paper: Figure {figure} reproduction inputs',
            'upload_type': 'dataset',
            'description': ''.join(body),
            'creators': [
                {'name': 'Kim, Min Cheol'},
            ],
            'license': 'cc-by-4.0',
            'version': commit,
            'related_identifiers': [
                {'identifier': PAPER_DOI, 'relation': 'isSupplementTo',
                 'scheme': 'doi', 'resource_type': 'publication-article'},
                {'identifier': REPO, 'relation': 'isSupplementTo', 'scheme': 'url'},
            ],
            'keywords': ['single-cell RNA-seq', 'memento', 'method of moments',
                         'differential expression', 'reproducibility'],
        }
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('figure', type=int, choices=sorted(FIGURES))
    args = parser.parse_args()
    json.dump(metadata(args.figure), sys.stdout, indent=2)
    sys.stdout.write('\n')


if __name__ == '__main__':
    main()
