# Figure 3 — HTEC response to interferon

Regenerates every panel of Figure 3 of
[Kim et al., *Cell* 2024](https://www.cell.com/cell/fulltext/S0092-8674(24)01144-9).

| Panel | Shows | Original code |
| --- | --- | --- |
| A | UMAPs: all cells by type, ciliated by stim and by time | `version2/figure_4/umaps.ipynb` |
| B | LFC to IFN-α against β / γ / λ | `version3/mean_var/mean_analyze.ipynb` |
| C | LFC heatmaps across five timepoints | `version3/mean_var/mean_analyze.ipynb` |
| D | ISG coexpression network over time | `classify_isg/select_isgs.ipynb` |
| E | Baseline variability, canonical vs non-canonical ISGs | `classify_isg/select_isgs.ipynb` |
| F | Tonic sensitivity, canonical vs the rest | `classify_isg/select_isgs.ipynb` |
| G | Change in variability vs change in mean | `classify_isg/select_isgs.ipynb` |

The original folders use the paper's earlier numbering — `version2/figure_4` and
`figure_5`, and a `figures/fig6/` output path — so the table above is the mapping that
matters.

## Requirements

Python with scanpy, pandas, scipy, scikit-learn, seaborn, matplotlib, networkx and xlrd.

Figure 3 uses **this repository's** memento package, not the object-oriented rewrite that
Figure 2 needs. No separate clone required.

## Data

```bash
python data_manifest.py check     # 33 files, 9.6 GB
python data_manifest.py link      # build the panel-organized tree the scripts read
```

`check --root DIR` validates a copy instead of the source volume; `bundle --root DIR`
writes a standalone directory that `FIGURE3_DATA` can point at.

Panel F needs supplementary Table S1E of Mostafavi et al., *Cell* 2016 (`mmc2.xls`),
staged on the volume at `hbec/external/mostafavi2016_mmc2.xls`. If it is missing,
download `mmc2.xls` from that paper and put it there.

## Run

```bash
python run_isg_clustering.py correlations   # ~4 min, gene-by-gene moments
python make_figure3.py                      # all panels + figure3.png
```

Individual panels can be plotted alone (`python panel_bc_mean_response.py`). Figures land
in `figures/` as pdf and png; intermediates in `intermediate/`.

`run_isg_clustering.py validate` is optional and draws nothing — it reports how closely
the ISG clustering reproduces.

## Checks built in

`panel_efg_variability.py` prints two numbers on every run and compares them to values
recorded in the original notebook: among canonical ISGs differentially expressed at
FDR < 0.01, the fraction also significant for variability is **0.778 under IFN-β and
0.394 under IFN-γ**. If those drift, something upstream has changed.

## Notes

- **The ISG gene lists are recovered from published data, not re-derived.** The original
  pickles are gone, so `isg_gene_lists.py` reads them from Supplementary Table 2 and the
  notebook's own stored outputs. Re-running the clustering does not reproduce the
  non-canonical modules, so it is a validation step rather than the source of truth.
- **One module is unrecoverable.** The non-canonical set was three clustering modules;
  two survive (133 genes), the third (46 genes) was never written down anywhere. Panels D
  and E therefore cover 133 of ~179 non-canonical genes, and panel D draws 205 of 251
  nodes. Direction and significance are unaffected.
- **Panels B and C use the 6-hour tests**, matching the caption. The notebook hardcodes
  the 3-hour files; `--timepoint 3` reproduces that.
- **Panels E and F group genes differently**, as in the original: E contrasts the two ISG
  modules, F contrasts canonical ISGs against the rest of the macrophage tonic table.
- **Panel D is drawn as a network**, per the caption; the notebook drew the same matrices
  as heatmaps, which are also written to `figures/figure3D_heatmaps.png`.
- The stored tests under `hbec/binary_test_*` were written by memento 0.0.6/0.0.9, whose
  result layout today's `get_1d_ht_result` cannot parse. `memento_legacy.py` reads them.
