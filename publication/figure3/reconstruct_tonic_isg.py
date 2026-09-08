"""Rebuild the `tonic_isg.txt` that panel 3F reads.

That file is not on either data volume, not in the repository, and never appeared in
the S3 sync log, so it is reconstructed from its source: supplementary Table S1E of

    Mostafavi et al., "Parsing the Interferon Transcriptional Network and Its Disease
    Associations", Cell 2016 (file mmc2.xls, authored by C. Benoist).

Sheet S1E holds two stacked blocks — B cells from row 3, macrophages from row 282.
Panel 3F uses the **macrophage** block. The original notebook read a tab-separated file
with a `GeneSymbol` column and a `Tonic Sensitivity` column, so `TonicIndex Macrophages`
is renamed to the latter here; `select_isgs.ipynb` then uppercases the symbols (the
source is mouse, the HTEC data human) and takes the log.

Emits the same columns the notebook consumed, so nothing downstream has to change.
"""

import os

import pandas as pd

import config

SOURCE = config.FIGURE3_DATA + 'panelDEFG_isg/external/mostafavi2016_mmc2.xls'
SHEET = 'S1E'
# Row indices into the raw, header-less sheet. The macrophage header sits at 281.
MACROPHAGE_HEADER_ROW = 281
MACROPHAGE_FIRST_ROW = 282
COLUMNS = ['ProbeSetID', 'GeneSymbol', 'IFN.FC.WT', 'Tonic Sensitivity', 'TonicIndex pval']


def build():
    if not os.path.exists(SOURCE):
        raise SystemExit(
            f'{SOURCE} is missing.\n\n'
            'Panel F needs supplementary Table S1E of Mostafavi et al., Cell 2016. It is\n'
            "a publisher's supplementary file, so it is not redistributed in the data\n"
            'bundle. Download mmc2.xls from\n'
            '  https://doi.org/10.1016/j.cell.2016.01.012\n'
            f'and save it as\n  {SOURCE}\n\n'
            'The rest of Figure 3 runs without it.')
    raw = pd.read_excel(SOURCE, sheet_name=SHEET, header=None)

    header = raw.iloc[MACROPHAGE_HEADER_ROW].astype(str).tolist()
    if 'Macrophages' not in ' '.join(header):
        raise ValueError(
            f'row {MACROPHAGE_HEADER_ROW} of sheet {SHEET} is not the macrophage header '
            f'({header}); the supplementary file layout has changed')

    table = raw.iloc[MACROPHAGE_FIRST_ROW:].copy()
    table.columns = COLUMNS
    table = table.dropna(subset=['GeneSymbol']).reset_index(drop=True)
    for column in ['IFN.FC.WT', 'Tonic Sensitivity', 'TonicIndex pval']:
        table[column] = pd.to_numeric(table[column], errors='coerce')
    return table.dropna(subset=['Tonic Sensitivity'])


def main():
    table = build()
    out = config.intermediate_path('tonic_isg.txt')
    table.to_csv(out, sep='\t', index=False)
    print(f'wrote {out}: {table.shape[0]} rows, {table["GeneSymbol"].nunique()} unique genes')
    print(f'Tonic Sensitivity range {table["Tonic Sensitivity"].min():.3f} '
          f'to {table["Tonic Sensitivity"].max():.3f}')


if __name__ == '__main__':
    main()
