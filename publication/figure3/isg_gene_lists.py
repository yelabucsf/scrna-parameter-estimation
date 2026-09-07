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

The notebook built its non-canonical set from three clustering modules. Two survive:
`noncanonical_genes_1` (72 genes, confirmed twice over -- by the supplementary table and
by a `print` in an earlier revision of this notebook named coexpression.ipynb) and
`noncanonical_genes_2` (61 genes, from cell 46's stored output). `noncanonical_genes_3`
(46 genes) was never printed in any of the 1109 notebook blobs in this repository's
history, and the 251-gene `all_selected_genes` it could be subtracted out of is likewise
gone -- so 133 of the original 179 non-canonical genes are recoverable.

The `module` column keeps the two apart, because they are not interchangeable:
Supplementary Table 2's differential-correlation analysis covers `noncanonical_1` only.
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
NONCANONICAL_2_LIST_CELL = 46


def _genes_by_type(table, isg_type):
    rows = table[table['type'] == isg_type]
    return set(rows['gene_1']) | set(rows['gene_2'])


def _notebook_gene_list(cell_index):
    """A gene list preserved in one of the notebook's stored cell outputs."""
    with open(NOTEBOOK) as handle:
        notebook = json.load(handle)
    for output in notebook['cells'][cell_index].get('outputs', []):
        text = ''.join(output.get('text', [])) or ''.join(
            output.get('data', {}).get('text/plain', []))
        if text.strip().startswith('['):
            return set(ast.literal_eval(text.strip()))
    raise ValueError(f'cell {cell_index} of {NOTEBOOK} no longer holds a gene list')


def load():
    """Gene, isg_class and module over the recoverable ISG modules."""
    table = pd.read_csv(SUPPLEMENTARY_TABLE)
    modules = {
        'canonical': _genes_by_type(table, 'canonical') | _notebook_gene_list(CANONICAL_LIST_CELL),
        'noncanonical_1': _genes_by_type(table, 'noncanonical'),
        'noncanonical_2': _notebook_gene_list(NONCANONICAL_2_LIST_CELL),
    }

    for left, right in [('canonical', 'noncanonical_1'), ('canonical', 'noncanonical_2'),
                        ('noncanonical_1', 'noncanonical_2')]:
        overlap = modules[left] & modules[right]
        if overlap:
            raise ValueError(
                f'{len(overlap)} genes are in both {left} and {right}: {sorted(overlap)[:5]}')

    rows = [(gene, 'canonical' if module == 'canonical' else 'noncanonical', module)
            for module, genes in modules.items() for gene in sorted(genes)]
    return pd.DataFrame(rows, columns=['gene', 'isg_class', 'module'])


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
