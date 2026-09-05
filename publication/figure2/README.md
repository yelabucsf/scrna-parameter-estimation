# Figure 2 — method validation and comparisons

Runnable reproduction of every panel of Figure 2 of
[Kim et al., *Cell* 2024](https://www.cell.com/cell/fulltext/S0092-8674(24)01144-9).

The original code for these panels is spread across
`publication/validation/estimation/` and `publication/validation/inference/` as a mix
of scripts and notebooks that hardcode paths (`/home/ubuntu/Data/`,
`/data_volume/memento/`, `/data_volume/bulkrna/`) which no longer exist. The scripts
here are ports of that code onto the current data volume, with the path handling
pulled out into `config.py`.

## Panels

| Panel | What it shows | Source of the original code |
| --- | --- | --- |
| A | Lin's concordance of mean / variability / correlation estimates against simulated ground truth, vs capture efficiency | `validation/estimation/simulation/{mean,variance,correlation}/` |
| B | Pearson correlation of Drop-seq estimates against smFISH, vs number of cells | `validation/estimation/smfish/{mean,variance,correlation}/` |
| C | Power vs FDR for DM, DV and DC | `validation/inference/simulation/{de,dv,dc}/` |
| D | Concordance AUC of single-cell vs pseudobulk DM against bulk RNA-seq | `validation/inference/bulk_comparison/` |
| E | Runtime vs number of cells | `validation/inference/runtime/` |

## Setup

Needs a Python environment with scanpy, pandas, scipy, scikit-learn, seaborn and
matplotlib. On this machine, `conda run -n antxr2` has all of them.

Two external pieces are pulled in:

* **The object-oriented memento rewrite** (`github.com/mincheoly/memento`), cloned to
  `/home/ubuntu/Github/memento`. Panels A and B import `memento.estimator` and
  `memento.auxillary.simulate` from it, matching the `sys.path.append` in the original
  scripts. Override with `MEMENTO_OO_PATH`.
* **The data volume**, `/memento_data/`. Override with `MEMENTO_DATA_PATH`.

One input file was missing from the data volume and had to be re-downloaded — the
Kang et al. interferon-stimulated PBMC dataset that panel A draws its simulation
parameters from:

```
curl -L -o /memento_data/interferon_filtered.h5ad \
  https://memento-examples.s3.us-west-2.amazonaws.com/pbmc-ifnb/interferon_filtered.h5ad
```

## Running

```bash
# Compute steps (~20 min and ~5 min respectively). Only these two need rerunning.
python panel_a_run_simulations.py all
python panel_b_run_correlation.py

# Plot every panel and assemble figures/figure2.png
python make_figure2.py
```

Individual panels can also be plotted on their own (`python panel_c_power_fdr.py`).
Estimates and per-panel summary tables land in `intermediate/`; figures land in
`figures/` as both pdf and png.

## What is recomputed and what is read off the volume

**Recomputed from raw inputs**

* Panel A, all three quantities. The simulations are cheap and none of the outputs
  were still on the volume. `panel_a_run_simulations.py` keeps everything in memory;
  the original variance script round-tripped each replicate through a pair of h5ad
  files so that BASiCS could pick them up in R.
* Panel B, correlation only — see the note below.

**Read off the volume**

* Panel B, mean and variability: `smfish/{mean,variance}/sample_*.npz`.
* Panel C: every method's output under `simulation/{de,dv,dc}/`.
* Panel D: every method's output under `canogamez/`, `hagai/` and `lupus_bulk/`.

**Neither** — Panel E is a set of wall-clock measurements transcribed from
`validation/inference/runtime/plots.ipynb`, replotted as-is.

## Deviations from the published panels

* **Panel A (variability) is missing the BASiCS curve.** BASiCS is an R package and
  there is no R installation on this machine; the per-replicate parameter csvs it
  produced are also not on the volume. The memento, Poisson and naive curves are
  regenerated. Panel B's BASiCS curve is unaffected because those numbers are baked
  into the stored npz.
* **Panel B (correlation) is recomputed rather than read from the volume.** The stored
  `smfish/correlation/sample_correlations.npz` holds one unnamed column per gene pair,
  ordered by whatever `smfish_estimates.npz['corr_genes']` contained when the estimates
  were produced. That order comes from a `set` intersection in
  `smfish/preprocess_fish.py`, so it is not stable between runs, and the reference file
  on the volume was regenerated at some later point. Scoring the stored estimates
  against the current reference gives *negative* correlations for every method.
  `panel_b_run_correlation.py` recomputes the memento, Poisson and naive estimates from
  the per-subsample h5ads against the current pair order, and looks up the stored SAVER
  and scVI results by gene name. The resulting values track the numbers cached in
  `smfish_correlation_comparison.ipynb` (memento 0.19 → 0.53 across 500 → 8000 cells).
* **Panel A (correlation) is drawn at 500 cells.** The published caption says 100, but
  `correlation_comparison.ipynb` — the notebook that produced the panel — used 500, and
  at 100 cells the spread across replicates swamps the separation between memento and
  the naive estimator. Change `PANEL_NUM_CELL` in `panel_a_plot.py` to see both.
* **memento's results are named `quasiGLM` on the volume**, not the `quasiML` that
  `bulk_comparison/plotting_utils.py` looks for, and they sit next to the other method
  outputs rather than in a local `temp/`.
* **Cano-Gamez needs an Ensembl ID → symbol map** (bulk results use Ensembl IDs, the
  single-cell results use symbols). The original `conversion.txt` is not in the repo, so
  `panel_d_bulk_concordance.py` fetches the mapping from Ensembl BioMart and caches it
  in `intermediate/ensembl_gene_symbols.tsv`.
* **`RNAHypergeometric.estimate_size_factor` no longer exists** in the memento rewrite.
  `memento_size_factor.py` reimplements it from `memento/main.py::setup_memento` in this
  repository, which still performs the same trimmed least-variable-gene computation.

## Things that stay stochastic

Panel A resimulates from scratch, so its curves will not be identical to the published
ones point for point — the seed used for the paper was not recorded. `--seed` makes a
given run reproducible. Panel B's mean and variability panels subsample the Drop-seq
data, but those subsamples are fixed files on the volume, so they are deterministic.
