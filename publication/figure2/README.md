# Figure 2 — method validation and comparisons

Runnable reproduction of every panel of Figure 2 of
[Kim et al., *Cell* 2024](https://www.cell.com/cell/fulltext/S0092-8674(24)01144-9).

The original code for these panels is spread across
`publication/validation/estimation/` and `publication/validation/inference/` as a mix
of scripts and notebooks that hardcode paths (`/home/ubuntu/Data/`,
`/data_volume/memento/`, `/data_volume/bulkrna/`) which no longer exist. The scripts
here are ports of that code, with path handling pulled out into `config.py` and the
data reorganized by panel (see below).

## Panels

| Panel | What it shows | Source of the original code |
| --- | --- | --- |
| A | Lin's concordance of mean / variability / correlation estimates against simulated ground truth, vs capture efficiency | `validation/estimation/simulation/{mean,variance,correlation}/` |
| B | Pearson correlation of Drop-seq estimates against smFISH, vs number of cells | `validation/estimation/smfish/{mean,variance,correlation}/` |
| C | Power vs FDR for DM, DV and DC | `validation/inference/simulation/{de,dv,dc}/` |
| D | Concordance AUC of single-cell vs pseudobulk DM against bulk RNA-seq | `validation/inference/bulk_comparison/` |
| E | Runtime vs number of cells | `validation/inference/runtime/` |

## Data organization

`/memento_data` is a flat sync of `s3://memento-paper/revision/`, organized by
*dataset*. That says nothing about which figure needs what, and Figure 2 turns out to
touch only **4.1 GB of the 329 GB volume**, scattered across nine top-level folders.

`data_manifest.py` is the single declarative inventory of every file Figure 2 depends
on, and drives three things:

```bash
python data_manifest.py check     # is every input present? prints per-panel counts and sizes
python data_manifest.py link      # build the panel-organized tree as symlinks (free, reversible)
python data_manifest.py bundle    # copy that tree into a standalone ~4 GB directory
```

The tree it builds is what every panel script actually reads (`config.FIGURE2_DATA`):

```
figure2_data/
  panelA_simulation/      interferon_filtered.h5ad, basics/
  panelB_smfish/          reference/ mean/ variance/ correlation/{subsamples,saver,scvi}
  panelC_inference/       dm/ dv/ dc/
  panelD_bulk/            canogamez/{bulk,single_cell}  hagai/…  lupus/…
```

Files are tagged `required` (read directly by a panel script) or `provenance` (not read
at plot time, but needed to regenerate a `required` file). `check` reports both.

The raw volume is left untouched, so the S3 correspondence still holds and none of the
not-yet-ported scripts for Figures 3–6 are broken. Because the scripts read the tree
rather than the volume, a `bundle` directory is sufficient on its own — set
`FIGURE2_DATA=/path/to/unpacked/bundle` and no 329 GB sync is needed.

## Setup

```bash
conda env create -f r-env.yml          # R 4.5 + BASiCS, for panel A only
python data_manifest.py check          # verify inputs
```

The Python side needs scanpy, pandas, scipy, scikit-learn, seaborn and matplotlib; on
this machine `conda run -n antxr2` has all of them.

Two external pieces are pulled in:

* **The object-oriented memento rewrite** (`github.com/mincheoly/memento`), cloned to
  `/home/ubuntu/Github/memento`. Panels A and B import `memento.estimator` and
  `memento.auxillary.simulate` from it, matching the `sys.path.append` in the original
  scripts. Override with `MEMENTO_OO_PATH`.
* **The Kang et al. interferon PBMC dataset**, which panel A draws its simulation
  parameters from. It was missing from the volume and had to be re-downloaded:

  ```bash
  curl -L -o /memento_data/interferon_filtered.h5ad \
    https://memento-examples.s3.us-west-2.amazonaws.com/pbmc-ifnb/interferon_filtered.h5ad
  ```

## Running

```bash
# Compute steps. Only these need rerunning; everything else is plotting.
python panel_a_run_simulations.py all --dump-basics-inputs   # ~20 min
Rscript run_basics_simulation.R                              # ~5 h, or shard by q (below)
python panel_b_run_correlation.py                            # ~5 min
python data_manifest.py link                                 # pick up the new BASiCS output

# Plot every panel and assemble figures/figure2.png
python make_figure2.py
```

The BASiCS step shards cleanly — each process writes a disjoint set of files and skips
any that already exist, so it is restartable:

```bash
for q in 0.05 0.1 0.2 0.3 0.5; do Rscript run_basics_simulation.R $q 3 & done
```

Individual panels can also be plotted on their own (`python panel_c_power_fdr.py`).
Estimates and per-panel summary tables land in `intermediate/`; figures land in
`figures/` as both pdf and png.

## What is recomputed and what is read off the volume

**Recomputed from raw inputs**

* Panel A, all three quantities, plus the BASiCS arm of the variability panel.
  `panel_a_run_simulations.py` keeps the replicates in memory; the original variance
  script round-tripped each one through a pair of h5ad files so BASiCS could pick them
  up in R. `--dump-basics-inputs` writes just the slice BASiCS scores, as MatrixMarket.
* Panel B, correlation only — see the note below.

**Read off the volume**

* Panel B, mean and variability: `smfish/{mean,variance}/sample_*.npz`.
* Panel C: every method's output under `simulation/{de,dv,dc}/`.
* Panel D: every method's output under `canogamez/`, `hagai/` and `lupus_bulk/`.

**Neither** — Panel E is a set of wall-clock measurements transcribed from
`validation/inference/runtime/plots.ipynb`, replotted as-is. Re-measuring BASiCS and
scHOT would need the runtime simulation datasets, which are not on the volume.

## Deviations from the published panels

* **Panel A's BASiCS arm is regenerated, not recovered.** The original
  `{num_cell}_{q}_{trial}_parameters.csv` files were never uploaded to
  `s3://memento-paper/revision/` — the bucket's `simulation/` prefix only ever held
  `dc/`, `de/` and `dv/` — and they are not on any volume here. Because the inputs are
  simulated rather than measured, they are regenerable, which is what
  `run_basics_simulation.R` does. It scores only the slice the published panel is drawn
  at (100 cells, q < 0.6): 100 MCMC runs rather than the 480 the original loops over.
* **`run_basics_simulation.R` drops Seurat and SeuratDisk.** The original used them only
  to read counts out of an h5ad. Reading MatrixMarket instead removes the most fragile
  dependency in that pipeline (`Convert()` on an h5ad) and cuts the R environment to
  BASiCS plus SingleCellExperiment.
* **Panel B (correlation) is recomputed rather than read from the volume.** The stored
  `smfish/correlation/sample_correlations.npz` holds one unnamed column per gene pair,
  ordered by whatever `smfish_estimates.npz['corr_genes']` contained when the estimates
  were produced. That order comes from a `set` intersection in
  `smfish/preprocess_fish.py`, so it is not stable between runs, and the reference file
  on the volume was regenerated at some later point. Scoring the stored estimates
  against the current reference gives *negative* correlations for every method.
  `panel_b_run_correlation.py` recomputes memento, Poisson and naive from the
  per-subsample h5ads against the current pair order, and looks up the stored SAVER and
  scVI results by gene name. The values then track the numbers cached in
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

## Known bad data on the volume

`simulation/dv/stim_chain.rds` is a byte-identical copy of `ctrl_chain.rds` (same md5,
2.95 GB each). `run_dv_basics.r` line 43 reads `saveRDS(ctrl_chain, file =
'stim_chain.rds')` — it saves the wrong object. `dv_basics.csv` itself is correct,
because `BASiCS_TestDE` was called with the right in-memory chain, so Figure 2C is
unaffected. But the saved chains are not a usable checkpoint: re-running the test from
them compares ctrl against ctrl. Neither file is in the manifest for that reason.

## Things that stay stochastic

Panel A resimulates from scratch, so its curves will not be identical to the published
ones point for point — the seed used for the paper was not recorded. `--seed` makes a
run reproducible, and each quantity is seeded independently so `panel_a_run_simulations.py
variance` alone reproduces what `all` produces. BASiCS adds its own MCMC sampling on top.
Panel B's mean and variability panels subsample the Drop-seq data, but those subsamples
are fixed files on the volume, so they are deterministic.
