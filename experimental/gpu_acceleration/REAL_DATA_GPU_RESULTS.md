# Public GPU backend: donor-adjusted interferon tests

The package now exposes the shared-cell bootstrap through `ht_1d_moments(backend="gpu")`, without compression or an executor monkeypatch. CPU remains the default.

## Design

`interferon_filtered.h5ad`: separate ctrl-versus-stim tests within each of eight cell types. Groups are donor × cell type × condition; donor indicators are covariates, with the existing regression intercept. Only complete donor pairs with at least 10 cells in each condition are included. Capture rate is 0.07; `min_perc_group=0.7`. Each cell type is normalized and filtered separately.

10,000 cell-bootstrap draws; primary seed 5, independent repeat seed 6. Normal ASL, hyper-relative moments, no replicate resampling. Inference uses the existing cell-bootstrap model conditional on these donors; it does not add donor-level resampling. BH correction is separate within each cell type and each moment family, not across the full combined table.

## Results

| Cell type | Cells | Donor pairs | Genes | GPU seconds (seeds 5 / 6) | DE FDR < .05 | DV FDR < .05 |
|---|---:|---:|---:|---:|---:|---:|
| CD14+ Monocytes | 5341 | 8 | 1742 | 3.72 / 3.38 | 1407 | 134 |
| B cells | 2564 | 8 | 969 | 3.47 / 3.44 | 411 | 6 |
| CD4 T cells | 10342 | 8 | 1482 | 3.22 / 2.80 | 746 | 10 |
| CD8 T cells | 2035 | 8 | 436 | 1.95 / 1.70 | 201 | 2 |
| Dendritic cells | 384 | 6 | 857 | 2.39 / 2.28 | 383 | 8 |
| FCGR3A+ Monocytes | 1586 | 8 | 1213 | 4.05 / 4.24 | 737 | 67 |
| Megakaryocytes | 144 | 4 | 343 | 0.65 / 0.61 | 12 | 2 |
| NK cells | 1988 | 8 | 625 | 2.55 / 2.43 | 319 | 2 |

All tested genes have finite DE and DV p-values. Positive coefficients indicate higher mean or residual variability in stim. Extremely small normal-tail probabilities can underflow to zero.

Dendritic cells exclude donors 107 and 1039; megakaryocytes exclude 101, 107, 1016, and 1039. All other cell types retain eight donors. Cell counts and inclusion flags are in [donor_cell_counts.csv](results_cell_types/donor_cell_counts.csv).

Public GPU calls totaled 21.98 seconds for seed 5 and 20.87 seconds for seed 6. These timings include sampling, moments, and regression, but exclude loading/preparation and validation. Peak allocated CUDA tensors were 1.11 GiB, excluding CUDA context/allocator reserve. Default 2048 MiB memory target; RTX 3060 12 GB; torch 2.9.0+cu128. All processes were restricted to logical CPUs 0 and 1. Other machine workloads can affect timings.

## Validation

- Entire package suite: **63 passed**, including 15 GPU tests. Tests cover multinomial weights, fixed-weight moment arithmetic, CPU/GPU regressions, seed reproducibility, gene-specific designs, result ordering, streaming fallback, and unsupported options.
- Original CPU pipeline: 16 expression-stratified genes per cell type, 128 total, with 10,000 independent draws. Across DE and DV, GPU/CPU SE ratios ranged from 0.964 to 1.031; no NaN mismatches. This is a distributional check, not identical RNG output.
- Real-data regression check: original CPU regressions applied to exactly the GPU log-bootstrap inputs for those 128 genes, with 2,000 draws. Maximum absolute difference across coefficients, SEs, and p-values: 1.86e-14.
- All genes were rerun with seed 6. Per-cell-type SE and coefficient comparisons are in report.json. Borderline FDR calls may vary with bootstrap seed.

## Reproduce and inspect

```bash
conda activate torch
python experimental/gpu_acceleration/run_cell_types.py
```

- [Combined primary results](results_cell_types/ctrl_vs_stim_1d.csv): coefficients, SEs, p-values and within-cell-type q-values.
- [Summary table](results_cell_types/summary.csv).
- [Full settings, timings and validation](results_cell_types/report.json).
- [Runner](run_cell_types.py); [package usage](../../README.md#optional-gpu-tests).

The backend supports hyper_relative, normal ASL, and resample_rep=False. This run covers 1D tests; the subsequent [correlation experiment](CORRELATION_RESULTS.md) validates the added GPU correlation path. Other estimator/ASL options remain unsupported. The existing size-factor approximation is retained. Shared weights preserve each gene’s bootstrap marginal distribution while changing dependence between genes relative to independent per-gene draws.
