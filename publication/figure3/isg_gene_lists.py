"""The canonical / non-canonical ISG lists that Figure 3 panels D-G depend on.

The originals lived in `canonical_isgs.pkl` and `noncanonical_isgs.pkl`, which are
gone. They are recovered here from two durable sources rather than re-derived, because
re-deriving them does not reproduce (see `run_isg_clustering.py validate`).

**Supplementary Table 2** is the primary source. It is the published output of
select_isgs.ipynb cell 100, which tagged every tested gene pair with the `type` column
-- 'canonical' or 'noncanonical' -- using exactly the two lists in question. Unpacking
its gene pairs recovers 70 canonical and 72 non-canonical genes. The 72 matches the
size the notebook's cell 42 reported for the non-canonical module, and contains HLA-A,
the marker cell 43 used to identify it.

**The notebook's own stored output** supplies the rest. Cell 40 printed all 72 canonical
genes verbatim; two of them (ADAR, TAP1) never entered the differential-correlation test
and so are absent from the table.

Note the non-canonical set here is the notebook's `noncanonical_genes_1` -- the only one
it ever saved to a pickle, and the one panels D, F and G use. Its `noncanonical_genes_2`
(61 genes) survives in cell 46's output but is not part of these panels;
`noncanonical_genes_3` (46 genes) was never printed and is not recoverable.
"""

import ast
import json
import os

import pandas as pd

import config

SUPPLEMENTARY_TABLE = config.DATA_PATH + 'tables/Supplementary_Table_2_HTEC_DC.csv'
NOTEBOOK = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'hbec_interferon', 'classify_isg', 'select_isgs.ipynb')
CANONICAL_LIST_CELL = 40


def _genes_by_type(table, isg_type):
    rows = table[table['type'] == isg_type]
    return set(rows['gene_1']) | set(rows['gene_2'])


def _notebook_canonical():
    """Cell 40's printed list, the only surviving record of the full canonical set."""
    with open(NOTEBOOK) as handle:
        notebook = json.load(handle)
    for output in notebook['cells'][CANONICAL_LIST_CELL].get('outputs', []):
        text = ''.join(output.get('text', [])) or ''.join(
            output.get('data', {}).get('text/plain', []))
        if text.strip().startswith('['):
            return set(ast.literal_eval(text.strip()))
    raise ValueError(f'cell {CANONICAL_LIST_CELL} of {NOTEBOOK} no longer holds a gene list')


def load():
    """Return a DataFrame of gene, isg_class over the canonical and non-canonical sets."""
    table = pd.read_csv(SUPPLEMENTARY_TABLE)
    canonical = _genes_by_type(table, 'canonical') | _notebook_canonical()
    noncanonical = _genes_by_type(table, 'noncanonical')

    overlap = canonical & noncanonical
    if overlap:
        raise ValueError(f'{len(overlap)} genes are in both ISG classes: {sorted(overlap)[:5]}')

    return pd.DataFrame(
        [(gene, 'canonical') for gene in sorted(canonical)]
        + [(gene, 'noncanonical') for gene in sorted(noncanonical)],
        columns=['gene', 'isg_class'])


def canonical_genes():
    classes = load()
    return classes.query('isg_class == "canonical"')['gene'].tolist()


def noncanonical_genes():
    classes = load()
    return classes.query('isg_class == "noncanonical"')['gene'].tolist()


def main():
    classes = load()
    out = config.intermediate_path('isg_classes_canonical.csv')
    classes.to_csv(out, index=False)
    counts = classes['isg_class'].value_counts().to_dict()
    print(f'wrote {out}: {counts}')
    print('canonical  :', ', '.join(classes.query("isg_class == 'canonical'")['gene'][:10]))
    print('noncanonical:', ', '.join(classes.query("isg_class == 'noncanonical'")['gene'][:10]))


if __name__ == '__main__':
    main()
