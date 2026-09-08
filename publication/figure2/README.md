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

Everything Figure 2 needs is published as one archive, 4.1 GB unpacked. No S3 access and
no data volume.

All five figures share one Zenodo record; download just this figure's archive.
DOI: _pending_

```bash
mkdir -p ~/memento_bundles && cd ~/memento_bundles
curl -L -O https://zenodo.org/records/<RECORD>/files/figure2_data.tar.gz
curl -L -O https://zenodo.org/records/<RECORD>/files/figure2_data.tar.gz.sha256
sha256sum -c figure2_data.tar.gz.sha256      # macOS: shasum -a 256 -c
tar -xzf figure2_data.tar.gz
export MEMENTO_DATA_PATH=~/memento_bundles
```

The archive unpacks to `figure2_data/`, organized by panel:

| Directory | Feeds |
| --- | --- |
| `panelA_simulation/` | A — including the BASiCS parameter estimates, already computed |
| `panelB_smfish/` | B |
| `panelC_inference/` | C |
| `panelD_bulk/` | D |

Panel E needs no data files; its runtimes are literals in `panel_e_runtime.py`.

## Run

```bash
python panel_a_run_simulations.py all      # ~20 min, the estimator simulation
python panel_b_run_correlation.py          # ~5 min
python make_figure2.py                     # all panels + figure2.png
```

No R needed: panel A's BASiCS arm takes about two hours to recompute, so its parameter
estimates ship in the archive. Regenerating them is a maintainer step — see
[MAINTAINING.md](../MAINTAINING.md).

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
