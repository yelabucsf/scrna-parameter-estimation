# Figure 2 — method validation and comparisons

Regenerates every panel of Figure 2 of
[Kim et al., *Cell* 2024](https://www.cell.com/cell/fulltext/S0092-8674(24)01144-9).

| Panel | Shows | Original code |
| --- | --- | --- |
| A | Lin's concordance of mean / variability / correlation estimates against simulated ground truth | `validation/estimation/simulation/` |
| B | Agreement of Drop-seq estimates with smFISH, vs number of cells | `validation/estimation/smfish/` |
| C | Power vs FDR for DM, DV and DC | `validation/inference/simulation/` |
| D | Concordance of single-cell vs pseudobulk DM with bulk RNA-seq | `validation/inference/bulk_comparison/` |
| E | Runtime vs number of cells | `validation/inference/runtime/` |

## Requirements

Python with scanpy, pandas, scipy, scikit-learn, seaborn, matplotlib. R only for panel A's
BASiCS curve:

```bash
conda env create -f r-env.yml
```

Panels A and B import the object-oriented memento rewrite, which is a **separate
repository** from this one:

```bash
git clone https://github.com/mincheoly/memento ~/Github/memento   # set MEMENTO_OO_PATH
```

## Data

```bash
python data_manifest.py check     # 914 files, 4.1 GB
python data_manifest.py link      # build the panel-organized tree the scripts read
```

`check --root DIR` validates a copy instead of the source volume. `bundle --root DIR`
writes a standalone 4 GB directory; point `FIGURE2_DATA` at it to run without the full
data volume.

One input is not on the volume and must be downloaded once:

```bash
curl -L -o /memento_data/interferon_filtered.h5ad \
  https://memento-examples.s3.us-west-2.amazonaws.com/pbmc-ifnb/interferon_filtered.h5ad
```

## Run

```bash
python panel_a_run_simulations.py all --dump-basics-inputs   # ~20 min
Rscript run_basics_simulation.R                              # ~2 h, or shard (below)
python panel_b_run_correlation.py                            # ~5 min
python data_manifest.py link                                 # pick up the BASiCS output
python make_figure2.py                                       # all panels + figure2.png
```

The BASiCS step shards cleanly and skips completed files, so it is restartable:

```bash
for q in 0.05 0.1 0.2 0.3 0.5; do Rscript run_basics_simulation.R $q 3 & done
```

Individual panels can be plotted alone (`python panel_c_power_fdr.py`). Figures land in
`figures/` as pdf and png; estimates and summary tables in `intermediate/`.

## Notes

- **Panel A's correlation plot uses 500 cells**, matching the notebook that produced it.
  The published caption says 100, but at 100 the replicate spread swamps the separation
  between methods. Change `PANEL_NUM_CELL` in `panel_a_plot.py` to compare.
- **Panel B's correlation estimates are recomputed, not read from the volume.** The stored
  `smfish/correlation/sample_correlations.npz` is ordered by a gene-pair list that was
  regenerated after the fact, so scoring it against the current reference gives negative
  correlations for every method. `panel_b_run_correlation.py` rebuilds them.
- **Panel E replots recorded wall-clock measurements.** Re-measuring BASiCS and scHOT
  needs an R install plus simulation datasets that are not on the volume.
- **Panel A resimulates from scratch**, so its curves will not match the published ones
  point for point; the paper's seed was not recorded. `--seed` makes a run reproducible.
- `simulation/dv/stim_chain.rds` on the volume is a duplicate of `ctrl_chain.rds` and is
  not usable as a checkpoint. `dv_basics.csv` is unaffected, so panel C is fine.
