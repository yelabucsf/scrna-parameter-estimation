# Figure 3 — HTEC response to interferon

Runnable reproduction of every panel of Figure 3 of
[Kim et al., *Cell* 2024](https://www.cell.com/cell/fulltext/S0092-8674(24)01144-9),
following the same layout as `publication/figure2/`.

The original code is in `publication/hbec_interferon/`, spread over three generations of
notebooks (`version1/`, `version2/`, `version3/`) plus `classify_isg/`. The folder names
lag the paper — `version2/figure_4` and `figure_5` use the old numbering, and the fig
path inside `select_isgs.ipynb` points at `figures/fig6/`. The mapping below was
recovered by grepping every `savefig` call in the tree.

## Panels

| Panel | What it shows | Source notebook | Its output name |
| --- | --- | --- | --- |
| A | UMAPs: all cells by type, ciliated by stim and by time | `version2/figure_4/umaps.ipynb` | — |
| B | LFC to IFN-α against β / γ / λ | `version3/mean_var/mean_analyze.ipynb` | `stim_lfc_scatter.png` |
| C | LFC heatmaps across five timepoints | `version3/mean_var/mean_analyze.ipynb` | `de_heatmap_all_tp.pdf` |
| D | ISG coexpression network over time | `classify_isg/select_isgs.ipynb` | `ifnb_coex_tps.png` |
| E | Baseline variability, canonical vs non-canonical | `classify_isg/select_isgs.ipynb` | `baseline_and_tonic.pdf` |
| F | Tonic sensitivity, canonical vs the rest | `classify_isg/select_isgs.ipynb` | `baseline_and_tonic.pdf` |
| G | Change in variability vs change in mean | `classify_isg/select_isgs.ipynb` | `de_vs_dv_canonical.png` |

## Setup

```bash
python data_manifest.py check     # 33 files, 9.6 GB
python data_manifest.py link      # build the panel-organized tree
```

Needs scanpy, pandas, scipy, scikit-learn, seaborn, matplotlib, networkx and xlrd; on
this machine `conda run -n antxr2` has all of them.

Unlike Figure 2, Figure 3 depends on **this repository's** memento package, not the
object-oriented rewrite — the notebooks call `setup_memento`, `compute_1d_moments` and
`compute_2d_moments`, which only exist here.

## Running

```bash
python run_isg_clustering.py correlations   # ~4 min, gene-by-gene moments
python make_figure3.py                      # all panels, then figures/figure3.png
```

`run_isg_clustering.py validate` is optional — it reports how far the clustering
reproduces, and is not needed to draw anything.

## Two files that had to be reconstructed

**`tonic_isg.txt`** (panel F) is not on either volume, not in the repo, and never
appeared in the S3 sync log. `reconstruct_tonic_isg.py` rebuilds it from supplementary
Table S1E of Mostafavi et al., *Cell* 2016 (`mmc2.xls`, staged on the volume at
`hbec/external/`). That sheet stacks two blocks — B cells from row 3, macrophages from
row 282 — and panel F uses the macrophage one. The script asserts the macrophage header
is where it expects before extracting, so a changed file fails loudly rather than
silently yielding B cells.

**The canonical / non-canonical ISG lists** (panels D–G) lived in `canonical_isgs.pkl`
and `noncanonical_isgs.pkl`, which are gone and were never committed. `isg_gene_lists.py`
recovers them from two durable sources:

* **Supplementary Table 2**, the published output of `select_isgs.ipynb` cell 100, which
  tagged every tested gene pair `canonical` or `noncanonical` using exactly those lists.
* **The notebook's own stored outputs** — cell 40 printed all 72 canonical genes, cell 46
  the 61 genes of the second non-canonical module.

The two agree: the 72 non-canonical genes unpacked from the table are identical to those
printed by `print(noncanonical_genes_1)` in `coexpression.ipynb`, an earlier revision of
the same notebook found in git history. Independent sources, same set.

## Why the clustering is not re-derived

Re-running the agglomerative clustering does not reproduce the modules, so it is a
validation step (`run_isg_clustering.py validate`) rather than the source of truth:

* **Canonical survives.** At the notebook's `distance_threshold=15`, all 72 published
  canonical genes fall in one cluster, plus 11 extra interferon genes (Jaccard 0.867).
* **Non-canonical does not.** The published 72 get split across two clusters (29 and 30
  genes); the best single cluster recovers 30. One recovered cluster matches the
  notebook's cluster 2 on *every* summary statistic — 61 genes, mean within-cluster
  correlation 0.430 against 0.428 — while sharing only 13 of its 61 genes. Same size,
  same coherence, different membership: the partition is degenerate at that level.

## Validation

Cells 123 and 124 of `select_isgs.ipynb` left behind two exact numbers: among canonical
ISGs differentially expressed at FDR < 0.01, the fraction also significant for
variability at FDR < 0.1. Both reproduce to three decimals — **0.778 under IFN-β and
0.394 under IFN-γ** — which checks the legacy result reader, the recovered gene lists
and the panel G logic in one shot. `panel_efg_variability.py` prints both on every run.

## Deviations from the published panels

* **`noncanonical_genes_3` is missing.** The notebook's non-canonical set was three
  clustering modules; two are recoverable (72 + 61 = 133 genes), the third (46 genes,
  cluster 8) was never printed in any of the 1109 notebook blobs in this repository's
  history. Panels D and E therefore cover 133 of the original ~179 non-canonical genes,
  and panel D draws 205 of 251 nodes. Subtracting the known modules from the 617-gene
  ISG universe does not recover it — only 251 of those 617 were ever assigned to a
  module, so the subtraction yields 412 genes, not 46.

  The conclusions are unaffected. Widening from 72 to 133 non-canonical genes left
  panel E's effect size flat (canonical median 2.194 vs 1.030 → 1.056) while the p-value
  fell from 6.2e-11 to 1.3e-14.
* **Panels E and F group genes differently**, which the caption glosses over. Panel E
  contrasts the two ISG modules against each other. Panel F's second group is every
  *other* gene in the macrophage tonic table — interferon-induced genes that are not
  canonical ISGs — because that table is the universe it has values for.
* **Panel B and C use the 6-hour tests.** `read_result()` in the notebook hardcodes the
  3-hour files, but the caption says 6 hours and the cell below it sets an unused
  `tp = '6'`. At 6h the classification gives 66 type-1 / 35 type-2 / 437 shared, against
  33 / 38 / 537 at 3h, so 6h produces the clearer type-1 block the panel highlights.
  `--timepoint 3` reproduces the notebook's literal behaviour.
* **Panel C does not reproduce the notebook's `row_order[300:]` trim** of the shared ISG
  block. That line was written against a larger set; against the 322 shared genes here it
  would leave 9 rows and erase the block the panel exists to show.
* **Panel D is drawn as a network**, per the caption. The notebook rendered the same
  matrices as heatmaps; that view is also emitted, as `figure3D_heatmaps.png`. One shared
  spring layout is used across all timepoints — recomputing per timepoint lets nodes
  drift and makes the change unreadable.
* **Panel A follows the caption, not the notebook.** `umaps.ipynb` predates the final
  figure and drew only two UMAPs, both over the whole dataset. The caption calls for
  three, with the ciliated cells zoomed. Control is drawn last in the stim panel so that
  its 739 cells stay visible under the four interferons.
* **`get_1d_ht_result` cannot read any of the stored tests.** Every h5ad under
  `hbec/binary_test_*` was written by memento 0.0.6/0.0.9, whose `uns['memento']['1d_ht']`
  layout differs; today's package raises `KeyError: 'test_genes'`. `memento_legacy.py`
  reads the stored arrays directly and renames them to the columns the notebooks used.

## Not reproduced

`simulation/dv/stim_chain.rds` on the volume is a byte-identical copy of
`ctrl_chain.rds` — `run_dv_basics.r` line 43 saves the wrong object. That affects
Figure 2C's provenance, not Figure 3, and is documented in `publication/figure2/README.md`.
