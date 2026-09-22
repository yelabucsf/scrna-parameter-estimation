# Probability-cache microbenchmark

**Probability-only caching does not provide a useful demonstrated speedup.**
The shared cache improves paired median timings by about 0.4% in the real-input
confirmation run, and less than 1% even when testing only small-mean draws.
The per-state cache is not consistently faster. Keep this implementation as a
standalone experiment; do not integrate it into the current bootstrap pipeline.

## What was implemented

Three standalone CUDA variants use the same binomial algorithm, precision,
Philox stream assignment, launch shape and output storage:

1. `direct`: compute `log1p(-min(p,1-p))` within each small-mean binomial draw.
2. `state_cache`: load that logarithm from one fp32 value per input state.
3. `shared_cache`: load it from a table indexed by the exact folded probability.

BTRS rejection sampling is unchanged in all three. The reference routines and
Philox engine are extracted from the installed PyTorch headers at compile time,
with source hashes recorded in each report. A fourth `header_reference` kernel
calls the unmodified binomial routine for exact-output checks. The cached
inversion routine is mechanically derived by replacing its logarithm calculation
with a supplied argument. All sampler kernels use 78 registers per thread.

`sampler_cache_cuda.py` uses NVRTC and the CUDA driver already in the torch
environment. No packages or toolkit were installed. Runtime compilation takes
about 0.6 seconds, excluded from sampling times. CUDA compilation does not enable
fast math. The Philox engine is the installed implementation, with host-only
normal-generation code removed and fixed-size array storage adapted for NVRTC.
Its stream assignment differs from the public `torch.binomial` kernel, which
processes multiple elements per thread; this is not a direct speed comparison
against that public API.

## Workload and measurement

The real trace is captured before draws in the existing conditional-binomial
bootstrap, including its existing binned cell-size factors. Each trace has 1,024
nonfinal state positions sampled from eligible states across all 16 donor/condition
groups, with 10,000 remaining-count observations per position. Probabilities are
constant within a state, but remaining counts vary across replicates.

The initial trace uses 64 expression-stratified genes and seed 715. Confirmation
uses 128 genes and seed 716. A separate shuffled stress case mixes individual
draws, deliberately removing the current pipeline's within-state coherence.
The small-mean-only subset excludes rows that ever enter BTRS in the trace.

Warmup precedes timing. Variant order cycles through all six permutations, with
12 paired rounds initially and 60 in confirmation. Each round uses the same RNG
seed across variants, and seeds advance between rounds. CUDA events measure
individual launches; their intervals can include host submission gaps. Other
workstation activity remains a source of variability. Do not interpret these
measurements as evidence of sub-percent improvements.

### Confirmation results

| Input layout | Draws per launch | Direct | Per-state cache | Shared cache |
|---|---:|---:|---:|---:|
| Real states, replicates kept together | 10,240,000 | 7.219 ms | 7.247 ms | 7.182 ms |
| Small-mean-only real states | 7,940,000 | 1.044 ms | 1.040 ms | 1.035 ms |
| Shuffled stress case | 1,048,576 | 2.400 ms | 2.400 ms | 2.405 ms |

Table entries are median launch times. Median paired direct/cache ratios are:

| Input layout | Per-state cache | Shared cache |
|---|---:|---:|
| Real grouped states | 0.9841x | 1.0042x |
| Small-mean only | 1.0039x | 1.0089x |
| Shuffled stress | 1.0000x | 0.9972x |

Ratios below one mean slower. Paired medians need not equal the ratio of the
separate median times. The initial grouped trace gives 6.502, 6.444 and 6.458 ms,
respectively, with paired ratios 1.0089x and 1.0036x. No meaningful advantage
emerges across the traces or layouts.

The confirmation grouped trace has 8,377,629 small-mean draws, 1,840,967 BTRS
draws and 21,404 deterministic draws. These frequencies are not timing shares.
The small-mean-only experiment shows that unchanged BTRS work is not the sole
reason the cache's measured benefit is small.

## Construction and memory

Full-inventory probability inputs are already resident for construction timing.
For 1,466,122 state positions:

- One fp32 logarithm per state occupies **5.59 MiB**. Allocation and calculation
  take a median **0.176 ms** in confirmation (first measured call: 0.303 ms).
- The shared table has **9,422 values**, totaling **36.8 KiB**. A straightforward
  int32 state-to-table mapping adds **5.59 MiB**, so table sharing does not itself
  reduce the total storage relative to one fp32 value per state.
- Shared construction, including sorting/deduplication, full-state mapping,
  trace mapping and logarithms, takes a median **1.647 ms** (first: 5.482 ms).
  Temporary allocations are additional to the retained sizes above.

The initial report's shared setup excludes the full-state mapping; use the
confirmation report for this more complete setup measurement. Setup covers the
whole inventory while launch timings cover a subset; adding them directly would
not represent the full bootstrap. Setup is small, but the warm sampling benefit
is already too small to justify integration. Compilation, transfers and future
pipeline integration costs cannot improve that result.

The earlier feasibility estimate used two fp64 fields per probability. This
experiment tests just one fp32 logarithm, matching the sampler's arithmetic.

## Correctness evidence

- Both cache variants and `direct` match the unmodified header reference
  **bit for bit** on the synthetic cases and all measured real-input layouts
  at identical RNG inputs.
- 31 synthetic binomial cases use 200,000 draws each, covering zero/one counts
  and probabilities, tiny probabilities, values near one, group-sized counts,
  and both sides of the small-mean/BTRS threshold.
- Prespecified mean and variance checks require errors below six Monte Carlo
  standard errors; grouped probability-mass checks require chi-square p-values
  at least 1e-6. The standalone reference and an independent `torch.binomial`
  control both pass. Maximum absolute mean/variance z-scores are 2.09/2.04;
  minimum grouped-PMF p-values exceed 0.039 across both implementations.
- Philox integer output agrees with a scalar reference for multiple streams,
  blocks and seeds, including the standard zero-counter/zero-key known answer.
  Changing the seed changes nondegenerate sample outputs.
- Capturing the real bootstrap trace checks exact count conservation.

These are standalone binomial checks. No custom multinomial kernel or new
regression path was integrated, and no full-call speedup is claimed. The broader
sampler validation plan still applies to any future replacement.

## Interpretation

Reuse and a small cache footprint do not guarantee that caching removes enough
work. This optimization saves one logarithm per small-mean draw, while random
generation and the loop's per-random-number logarithms remain. It adds a memory
load, and shared lookup adds indexing. The measurements do not isolate which
of those costs explains the near-zero net change.

Stop this probability-only integration path. Count-dependent BTRS setup caching
remains untested: it removes a different collection of calculations and requires
its own correctness and performance experiment. These results do not establish
that it would help.

## Reproduce

```bash
conda activate torch
python experimental/gpu_acceleration/bench_sampler_cache.py
python experimental/gpu_acceleration/bench_sampler_cache.py \
  --genes 128 --seed 716 --repeats 60 \
  --out experimental/gpu_acceleration/results_sampler_cache_bench_confirm
```

Both scripts use at most two logical CPUs. The source dataset and memento's
package implementation remain unchanged. JSON reports record source hashes,
validation details, trial ordering and individual timings. NPZ trace files are
local, git-ignored artifacts.

Reports: [initial](results_sampler_cache_bench/report.json),
[confirmation](results_sampler_cache_bench_confirm/report.json).
