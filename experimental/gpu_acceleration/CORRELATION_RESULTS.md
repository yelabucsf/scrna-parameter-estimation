# GPU differential correlation

`memento.ht_2d_moments(..., backend="gpu")` now uses shared cell weights for
the bootstrap and CUDA for regression. CPU remains the default. This extends
the public 1D backend and does not use a monkeypatch or state compression.

## Real-data run

CD14+ Monocytes from `interferon_filtered.h5ad`: 5,341 cells, eight donors,
16 donor × cell-type × condition groups, 1,742 filtered genes. We tested stim
versus ctrl with donor covariates, capture rate 0.07, and
`min_perc_group=0.7`. Normal ASL; no replicate resampling. The inference is
conditional on these donors under the existing cell-bootstrap model.

A reproducible panel of **2,000 unordered, nonself pairs** was sampled uniformly
without replacement (selection seed 29) from 1,516,411 possible pairs. This is
not an all-pairs run. Each test used 10,000 bootstrap draws.

| Workload | GPU seconds | Compressed CPU seconds |
|---|---:|---:|
| 2,000 pairs, seed 5 | 5.94 | Not run |
| 2,000 pairs, seed 6 | 5.63 | Not run |
| Same 64-pair reference subset | 0.457 | 57.16 |

Timings include sampling, moments, and regression. Loading and observed-moment
preparation took 1.12 seconds separately. CUDA was initialized before timing;
the first CPU call includes worker startup. The 64-pair comparison is a single
measurement under machine load, not an extrapolated full-workload speedup.
All processes were restricted to two logical CPUs. Hardware: RTX 3060 12 GB;
torch 2.9.0+cu128. Peak live GPU tensors were **0.61 GiB**, excluding CUDA
context and allocator reserve, with the default 2048 MiB working-memory target.

All 2,000 pairs returned finite results. BH correction across this panel gave
8 discoveries at FDR < 0.05 with seed 5 and 9 with seed 6. These q-values apply
to the tested panel, not the complete set of possible gene pairs.

## Validation

- **66 package tests passed.** Added tests compare fixed-weight covariance and
  correlation calculations with the existing CPU estimators, verify masked and
  pair-specific regressions, invalid-draw replacement, ordering, diagonal NaNs,
  repeatability, empty selections, and the memory-limited streaming path.
- Original compressed CPU pipeline: 64 pairs spanning baseline correlation
  ranks, 10,000 independent draws. GPU/CPU SE ratios: **0.978–1.024**, median
  1.0004; no NaN mismatches. Observed coefficients differed by at most
  **2.22e-16**.
- Original CPU regression applied to identical GPU bootstrap inputs for those
  64 pairs, with 2,000 draws: maximum absolute difference across coefficients,
  SEs and p-values **2.55e-15**.
- Independent full GPU repeat: SE ratios **0.961–1.039**, with the central 90%
  between 0.982 and 1.017; observed coefficients were identical. Borderline
  significance calls can vary with the bootstrap seed.

## Implementation details

For each cell resample, the same weight matrix generates the two means,
two noise-corrected second moments, and cross moment `x*y/sf**2`. Covariance
is the cross moment minus the product of means; correlation divides by the
square root of the two corrected variances. Cell sizes therefore enter both
marginal and cross moments, just as in the compressed reference.

Bootstrap draws with nonpositive variance or correlation outside `(-1, 1)`
are replaced by randomly chosen valid draws from the same pair/group. Groups
with invalid or boundary observed correlations, or no valid bootstrap draws,
are excluded. This preserves the reference handling. Raw correlations are
regressed without a Fisher or log transform. Unlike the 1D summary, the
reported correlation coefficient uses the observed estimate, matching CPU.

Weights are shared across pairs, cached when they fit, and regenerated in
chunks otherwise. Moment arithmetic uses fp32 with TF32 disabled; regression
uses fp64. The existing size-factor approximation remains in place. Pair
batching limits working memory; this first implementation recomputes marginal
moments for genes repeated across pairs rather than caching every gene's
bootstrap moments globally.

The supported scope is `hyper_relative`, `approx="norm"`, and
`resample_rep=False`. Observed-moment preparation and regression design
projections remain on CPU. Self-pairs return NaN. GPU pair-specific treatment
and covariate selections are supported.

## Reproduce and inspect

```bash
conda activate torch
python experimental/gpu_acceleration/run_correlations.py
```

- [Primary results](results_correlations/gpu_seed5.csv)
- [Independent repeat](results_correlations/gpu_seed6.csv)
- [Compressed CPU subset](results_correlations/cpu_subset.csv)
- [Pair panel](results_correlations/pairs.csv)
- [Settings and validation](results_correlations/report.json)
- [Runner](run_correlations.py) and [public API usage](../../README.md#optional-gpu-tests)
