# GPU experiments

The [fibroblast tissue comparison](FIBROBLAST_RESULTS.md) validates the full
9,487-gene panel on GPU and compressed CPU (10 workers), using bladder versus
subcutaneous adipose fibroblasts from five paired donors.

The [correlation extension](CORRELATION_RESULTS.md) adds GPU bootstrap and regression
through `ht_2d_moments(..., backend="gpu")`.

The shared-cell sampler is now available through the public
`memento.ht_1d_moments(..., backend="gpu")` API. See
[the eight-cell-type real-data run](REAL_DATA_GPU_RESULTS.md) for results,
validation, and the reproducible command. This backend does not monkeypatch
the executor; the older runners below remain experimental comparisons.

The latest [shared cell-weight experiment](CELL_MATMUL_RESULTS.md) completes the
full GPU call in 5.1–5.7 seconds, versus 17.8–18.2 seconds with compression.
It includes CPU/GPU comparisons, group-size sweeps, and estimator/regression checks.

This directory contains the experimental predecessors to the package backend.
Measured results and interpretation are in [RESULTS.md](RESULTS.md).
The follow-up on larger batches and sampler memory is in [BATCH_RESULTS.md](BATCH_RESULTS.md).
Cross-group batching, graph-cache experiments, and the next optimization targets
are in [MEMORY_SPEED_RESULTS.md](MEMORY_SPEED_RESULTS.md).
GPU state preparation reduces the full call to about 20.5 seconds; see
[GPU_STATE_RESULTS.md](GPU_STATE_RESULTS.md). The custom sampler remains a
separate proposed experiment with a [validation plan](SAMPLER_VALIDATION_PLAN.md).
The initial [sampler cache probe](SAMPLER_CACHE_PROBE.md) measures input reuse
and memory feasibility. The follow-up [CUDA microbenchmark](SAMPLER_CACHE_RESULTS.md)
finds no useful speedup from probability-only caching; it remains standalone.
The original runner exercises `memento.ht_1d_moments` by replacing its task executor
inside a scoped context. The CPU implementation is unchanged; PyTorch is now an optional package extra. Do not use the context concurrently with another memento call.

## Dataset and design

- `/mnt/c/Data/memento_workspace/interferon_filtered.h5ad`
- CD14+ Monocytes: 5,341 cells, eight donors, control/stimulated pairs.
- Groups: donor × stimulation; treatment: stimulated versus control;
  covariates: donor dummy variables (the regression fits an intercept).
- Capture rate `q=0.07`, as in the repository's interferon analysis.
- Normalization and moment fitting use all genes in this cell population.
  Default expression/variance filters with `min_perc_group=0.7` retain 1,742
  genes. Benchmark subsets sample evenly across the mean-expression ranking;
  they do not recompute normalization or fits on the smaller set.

The recorded experiments used two logical CPUs unless a report specifies ten.
Resource limits for automated tests now live in `tests/conftest.py`; use
`python -m pytest tests -q --cpu-limit=10`. The old `runtime.py` is removed.
Standalone runners inherit their resource settings from the shell; see
[testing instructions](../../tests/README.md) for a Linux `taskset` example.
GPU commands need access outside the sandbox in this WSL environment. The
existing `torch` conda environment was used.

CSV tables, PNG previews, and prepared datasets are generated locally by the
runners and excluded from git. JSON summaries, SVG plots, and validation reports
are included for review.

## Commands

From the repository root:

```bash
conda activate torch
python experimental/gpu_acceleration/real_data_bench.py \
  --genes 128 --boots 10000 --batch 128 \
  --modes cpu cpu_repeat gpu_cpu gpu_cpu_map gpu

python experimental/gpu_acceleration/validate_gpu.py \
  --prepared experimental/gpu_acceleration/results/prepared.pkl \
  --out experimental/gpu_acceleration/results/validation.json

python experimental/gpu_acceleration/real_data_bench.py \
  --genes 32 --boots 10000 --validate --modes gpu \
  --out experimental/gpu_acceleration/results_regression

python experimental/gpu_acceleration/real_data_bench.py \
  --genes 1742 --boots 10000 --batch 256 --modes gpu \
  --out experimental/gpu_acceleration/results_full256

python experimental/gpu_acceleration/real_data_bench.py \
  --genes 1742 --boots 10000 --batch 1024 --modes gpu \
  --out experimental/gpu_acceleration/results_full1024

python experimental/gpu_acceleration/plot_results.py \
  experimental/gpu_acceleration/results/report_128g_10000b.json

# Repeated full-workload gene-batch sweep, reversing order on alternate passes.
python experimental/gpu_acceleration/batch_sweep.py

# Check the sampler workspace separately from the gene-batch size.
python experimental/gpu_acceleration/batch_sweep.py \
  --batches 1024 1742 --sampler-mib 256 \
  --out experimental/gpu_acceleration/results_batch_sweep256/report.json
```

Modes:

| Mode | Bootstrap | Regression |
|---|---|---|
| `cpu` | Existing CPU implementation, seed 5 | Existing sklearn/numpy |
| `cpu_repeat` | Existing CPU implementation, seed 6 | Existing sklearn/numpy |
| `gpu_cpu` | Fused CUDA conditional binomial | Existing CPU, distributions copied back |
| `gpu_cpu_map` | Same CUDA draws | Factored regression on CPU, distributions copied back |
| `gpu` | Same CUDA draws | Factored regression and summaries on CUDA |

`--validate` checks regression on identical real bootstrap inputs and times the
reference CPU, factored CPU, resident GPU fp64, and resident GPU fp32 variants.
These extra checks are **included in public-call wall time**, so validation runs
must not be used as end-to-end performance measurements. Their regression
microtimings are recorded separately. CUDA is synchronized for timings and
initialized before the public calls. Preparation/loading is reported separately.
CPU first-call timings include joblib worker startup. Other processes on this
workstation can affect timing; these are measurements under load.

`batch_sweep.py` loads the same prepared dataset, warms all stages on a small
real-data call, and times every selected batch size twice by default. It clears
unused CUDA allocations before each trial, reports both peak live tensor and
allocator-reserved memory, and checks all output summaries are finite. Its
optional sampler-budget override is scoped to the experiment; package defaults
remain unchanged.

## Implementation and limits

`gpu_1d.py` compresses states using the current CPU `_unique_expr`, buckets state
counts in bands of 64, and accumulates moments without allocating a weight cube.
Conditional probabilities use float64 ratios of integer suffix sums, then convert
to fp32. Each gene's final real state has conditional probability one; padded
states have zero probability. Counts are restricted to at most `2**24` cells.
The existing `_get_batch_size` 64 MiB rule bounds live sampler arrays. It does
not bound retained distributions, regression temporaries, or the CUDA context;
the gene batch size controls those. Peak allocated tensor memory is reported.

Sampling/moment accumulation is fp32. Residual variance, logs, regression and
normal-tail summaries use fp64. The small weighted regression design is factored
on the CPU once per distinct valid-group mask, then applied to all draws on the
selected device. This retains sklearn's intercept and memento's marginal
treatment-coefficient semantics. Invalid draws are resampled uniformly from
valid draws. Only small masks, designs, and result summaries cross PCIe in the
resident path; full distributions cross for explicit CPU comparisons.

Supported experimental scope: `hyper_relative`, common treatment/covariate
design, `resample_rep=False`, normal ASL. Other estimators, gene-specific designs,
replicate resampling, generalized-tail ASL, and correlation are not implemented.
GPU RNG does not reproduce CPU draws. The experiment uses a fixed torch seed to
allow identical-input comparisons between the hybrid and resident GPU paths.

Outputs are JSON timing/validation reports and NPZ summaries. Local prepared
pickle caches and NPZ outputs are git-ignored. Only load caches generated by this
runner. The source dataset is read-only throughout.
