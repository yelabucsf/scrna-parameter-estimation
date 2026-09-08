"""Generate the Zenodo record metadata for the figure reproduction bundles.

    python zenodo_metadata.py > metadata.json

One record holds all five archives. Zenodo serves files individually, so a reader still
downloads only the figure they want; a single DOI is simply what a paper's reproduction
data is normally cited by.

This is written to be uploaded as a **new version of the existing memento record**, which
previously held code only. See MAINTAINING.md for the sequence.
"""

import argparse
import json
import subprocess
import sys

REPO = 'https://github.com/yelabucsf/scrna-parameter-estimation'
# TODO: confirm before uploading. Taken from the article page rather than from Crossref,
# and a wrong identifier would be baked into an immutable record.
PAPER_DOI = '10.1016/j.cell.2024.09.045'

# figure -> (subject, archive size, unpacked size)
FIGURES = [
    (2, 'method validation and comparisons', '1.3 GB', '4.1 GB'),
    (3, 'interferon stimulation in human airway epithelium', '2.7 GB', '9.6 GB'),
    (4, 'Perturb-seq of transcription factor knockouts', '1.6 GB', '4.9 GB'),
    (5, 'eQTL, vQTL and coexpression QTL analysis', None, '21 GB'),
    (6, 'memento in CZI CELLxGENE Discover', '44 MB', '44 MB'),
]

CAVEATS = [
    ('Figure 3, panel F', 'additionally needs supplementary Table S1E of Mostafavi et '
     'al., Cell 2016 (mmc2.xls). That is a publisher supplementary file and is not '
     'redistributed here; the figure README gives the DOI to fetch it from.'),
    ('Figure 5, panels F-I', 'additionally need individual-level genotypes, which are '
     'controlled-access under dbGaP phs002812.v1.p1 and are not included. Panels A-E are '
     'complete: the minor-allele-frequency filter panel A applies ships as an aggregate '
     'per-variant frequency, which contains no individual-level data.'),
    ('Figure 6, panels C and D', 'additionally query the public CELLxGENE census at run '
     'time, so that figure needs network access.'),
]


def git_commit():
    try:
        return subprocess.run(['git', 'rev-parse', '--short', 'HEAD'],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return 'unknown'


def description():
    rows = '\n'.join(
        f'<tr><td>figure{n}_data.tar.gz</td><td>Figure {n} — {subject}</td>'
        f'<td>{archive or "—"}</td><td>{unpacked}</td></tr>'
        for n, subject, archive, unpacked in FIGURES)
    caveats = '\n'.join(f'<li><strong>{who}</strong> {what}</li>' for who, what in CAVEATS)
    return f'''\
<p>Input data to reproduce the figures of Kim et al., <em>Cell</em> 2024, alongside the
memento source code.</p>

<p>Each archive holds one figure's inputs, organized by panel. Download only the figure
you need — the archives are independent.</p>

<table>
<tr><th>File</th><th>Figure</th><th>Download</th><th>Unpacked</th></tr>
{rows}
</table>

<p>Unpack an archive and point <code>MEMENTO_DATA_PATH</code> at the directory containing
it, then run that figure's script:</p>

<pre>tar -xzf figure3_data.tar.gz
export MEMENTO_DATA_PATH=$PWD
cd publication/figure3 &amp;&amp; python make_figure3.py</pre>

<p>Verify each download against its <code>.sha256</code> file before unpacking. Full
instructions, and what each figure should produce, are in
<a href="{REPO}/tree/main/publication">publication/</a>.</p>

<p>Three figures need something beyond their archive:</p>
<ul>
{caveats}
</ul>
'''


def metadata():
    return {
        'metadata': {
            'title': 'memento: source code and figure reproduction inputs',
            'upload_type': 'dataset',
            'description': description(),
            'creators': [{'name': 'Kim, Min Cheol'}],
            'license': 'cc-by-4.0',
            'version': git_commit(),
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
    argparse.ArgumentParser(description=__doc__).parse_args()
    json.dump(metadata(), sys.stdout, indent=2)
    sys.stdout.write('\n')


if __name__ == '__main__':
    main()
