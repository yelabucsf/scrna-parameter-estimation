# GPU state preparation

Moving state counting and coefficient preparation to GPU reduces the full
`ht_1d_moments` call by about 14%, with identical final outputs at the same seed.
The existing PyTorch conditional-binomial sampler and GPU regression are unchanged.

## Measurements

CD14+ Monocytes, 5,341 cells, 16 donor/condition groups, 1,742 genes,
10,000 bootstrap draws. RTX 3060 12 GB, existing torch environment, affinity
limited to two logical CPUs. Outer gene batch 256, sampler rows 1,024,
sampler workspace budget 256 MiB. Trials run sequentially in the order below.

| Preparation | Full call | Compression stage | Peak live GPU tensors |
|---|---:|---:|---:|
| CPU reference | 24.084 s | 4.479 s | 1.633 GiB |
| GPU | 20.674 s | 1.889 s | 1.636 GiB |
| GPU repeat | 20.284 s | 1.653 s | 1.636 GiB |
| CPU reference repeat | 23.786 s | 4.340 s | 1.633 GiB |

Average full-call reduction is 14.4%; compression falls from about 4.41 to
1.77 seconds. GPU allocator-reserved memory peaks at 5.084 GiB. Live/reserved
tensor measurements do not include every CUDA context allocation. Other
workstation processes can affect timings.

Shared dataset loading, normalization and fitting are excluded. Per-call task
assembly, state preparation, transfers, bootstrap, transformations and regression
are included. The sampler still takes about 13.7 seconds, and remains the largest
measured stage. Larger allocations alone have not demonstrated further gains.

## Implementation and limits

`gpu_states.py` uses integer keys for gene, size-factor code and expression,
then counts and decodes unique states on GPU. State order matches `_unique_expr`.
Probabilities and moment coefficients are prepared on GPU and retained there
until sampled. The dispatcher reuses the existing eager sampling loop.

Sparse task construction, bulk densification, input validation and size-factor
coding remain on CPU. Inputs are prepared per gene chunk; this is not yet a
whole-dataset resident cache. This experiment requires finite nonnegative integer
expression counts and rejects integer-key overflow. No package backend or default
behavior is changed.

## Validation

- All 27,872 real gene/group pairs match CPU state expressions and counts exactly.
  Every prepared sampling/moment coefficient is bit-for-bit identical on this dataset.
- Synthetic cases cover zero/constant genes, one-cell groups, several size-factor
  dtypes, noninteger size factors, large expression values and invalid inputs.
  General coefficient comparisons allow small floating-point rounding differences.
- A same-RNG bootstrap comparison gives zero mean/variance differences.
- Both full GPU-preparation trials match every final summary array from the
  CPU-preparation/GPU-bootstrap/GPU-regression run exactly at the same seed.
  This preserves the previously validated regression path; it is not a claim
  that CPU and GPU random-number generators produce the same draws.
- Existing mixed-row sampler checks pass in the benchmark harness.

Raw reports: [timings](results_gpu_states/report.json),
[state validation](results_gpu_states/validation.json).

## Reproduce

```bash
conda activate torch
python experimental/gpu_acceleration/validate_gpu_states.py
python experimental/gpu_acceleration/memory_speed_bench.py \
  --modes wide gpu_states gpu_states wide \
  --out experimental/gpu_acceleration/results_gpu_states
```

Both runners enforce the two-CPU limit. GPU execution requires the authorized
host GPU access in this environment.
