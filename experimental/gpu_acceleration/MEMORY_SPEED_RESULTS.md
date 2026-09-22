# Using memory to reduce work, not just retain more results

Cross-group batching improves the complete 1,742-gene, 10,000-bootstrap call
from 28–29 seconds to about 24.5 seconds. Caching CUDA graphs uses more memory
but provides no material further improvement in the tested implementation.
The next substantial targets are GPU-native state preparation and the binomial
sampler itself. The two-CPU affinity/thread limits remain enforced throughout.

## What changed

The previous dispatcher sampled one donor/condition group at a time. Increasing
the gene batch primarily increased the retained distributions and regression
temporaries. `WideGPUDispatch` instead combines independent **gene–group pairs**
from all 16 groups, buckets them by state count, and samples up to 1,024 rows
at once. Every row retains its own cell count, capture rate, compressed states,
and mean–variance fit. Biological groups are not pooled statistically.

The tested configuration retains 256 genes per outer batch, allows a 256 MiB
sampler workspace, and runs regression on GPU in fp64. It retains the existing
`_get_batch_size` rule, with a scoped workspace-budget override. Sampling and
moment accumulation stay fp32 with the same conditional-binomial algorithm.
Package source and default behavior are unchanged.

Two graph variants cache sampler operations and persistent buffers:

- `graph`: powers-of-two row counts, replaying blocks of 64 states. It reuses
  nine graphs across this workload, at the cost of extra padding.
- `graph_exact`: captures each exact row/state shape, retaining 41 graphs and
  more private workspace to avoid padding.

Graph replay reads updated probability/coefficient buffers and advances CUDA
RNG state. Returned samples have independent storage. PyTorch documents graph
capture/replay, RNG support, and graph-private memory pools in its
[2.9 CUDA documentation](https://docs.pytorch.org/docs/2.9/notes/cuda.html#cuda-graphs).

## Full-workload measurements

Same prepared dataset, model, expression-stratified gene order, and 10,000 draws
as earlier experiments. Shared data preparation is excluded; per-call state
compression, transfers, sampling, transformations, regression and public-call
task assembly are included. Graph cold-call timing includes capture costs.

| Implementation | Public-call wall time | Peak live GPU tensors |
|---|---:|---:|
| Original GPU dispatcher | 29.05, 28.28 s | 1.51 GiB before graph caches |
| Cross-group batching | 24.43, 24.53 s | 1.63 GiB before graph caches |
| Padded/block graph cache | 25.85 s cold; 25.19 s warm | 1.83–1.84 GiB |
| Exact-shape graph cache | 29.88 s cold; 24.46 s warm | 4.17–4.46 GiB |

The separate exact-shape experiment brackets its graph runs with ordinary
cross-group calls at 24.92 and 24.84 seconds. Exact-shape capture takes 5.58
seconds on the cold call; the warm call captures nothing. There is no useful
demonstrated speed advantage over cross-group batching alone.

These are timings under workstation load, not an isolated hardware benchmark.
Persistent graph caches remain resident during later modes in the same process,
so later eager-mode peak memory includes idle graph allocations. The table uses
the eager modes' pre-cache measurements for their standalone footprints.

For the original dispatcher, bootstrap takes about 17.7 seconds; cross-group
batching reduces this to about 14.5 seconds. State compression remains about
4.5 seconds. Transformations take about 1.3 seconds and regression about two
seconds. Additional GPU capacity cannot by itself remove those operations.

Reports: [original / cross-group / block graph](results_memory_speed/report.json),
[exact-shape cache](results_memory_exact/report.json).

## Sampler timing

A representative real-state batch has 256 rows, 778 cells per row, up to 124
states, and 10,000 draws. Three CUDA-event measurements assign 220.6–222.0 ms to
binomial calls and 36.7–37.2 ms to remaining-count/moment updates: **about 86%
of these timed loop intervals is in binomial sampling**.

These event intervals include host launch gaps and exclude preparation,
transfers and final normalization. They are not kernel-only CUPTI measurements,
nor a profile of every gene in the dataset. A CUPTI profiling attempt stalled
in an I/O wait in this environment and was terminated; no claims use that
attempt. The event measurements and lack of a useful graph-cache speedup make
the sampler a better next target than allocating larger result buffers.

Raw [event timings](results_memory_speed/profile.json).

## Correctness

- Mixed-cell-count and mixed-capture-rate sampler cases pass count conservation,
  marginal covariance checks in the small cases, and CPU/GPU bootstrap SD-ratio
  checks, including single-state rows and state counts crossing graph blocks.
- Both graph variants produce new random draws on successive replays. Returned
  block-graph samples remain unchanged after the cache is reused.
- Cross-group and block-graph regression on identical real bootstrap inputs
  agree with the CPU reference to maximum absolute differences of `2.64e-14`
  and `2.68e-14`, respectively, in the 32-gene correctness run.
- All ten full-workload trials return finite values for every gene and all six
  summaries. Median mean/variability SE ratios versus the existing CPU subset
  remain near one: approximately 0.9984–1.0016 across implementations.
- With a warm exact-shape cache and the same torch seed, every full-workload
  summary matches the eager cross-group result **bit for bit** in the tested
  run. Cold capture consumes warmup RNG draws, so cold-run equality is not
  expected; distributional checks apply there.

Reports: [small correctness run](results_memory_smoke/report.json),
[CPU-overlap checks](results_memory_speed/accuracy.json).

## Next optimizations, in priority order

Follow-up: the first GPU state-preparation implementation is now measured at
20.3–20.7 seconds per full call, with identical same-seed outputs. See
[GPU_STATE_RESULTS.md](GPU_STATE_RESULTS.md). The roadmap below records the
original proposals; whole-dataset input caching is still outstanding. Custom
sampling remains separate under [this validation plan](SAMPLER_VALIDATION_PLAN.md).

1. **GPU-native state preparation with reusable inputs.** Upload dense integer
   counts and size-factor codes once, then form/count expression states in
   batches on device. The eligible count matrix is only about 36 MiB in int32;
   there is ample room for histograms, compressed-state buffers, probabilities
   and moment coefficients. This targets the measured 4.5-second compression
   stage and could bypass much of the public wrapper's sparse-column task
   construction. No speedup has yet been measured for this redesign.
2. **A specialized exact multinomial kernel.** The current implementation avoids
   storing the weight cube, but still calls a generic binomial kernel and
   separate moment-update kernels for every state. A custom kernel could retain
   RNG state, remaining counts and moments across states. Use spare VRAM for
   reusable probability-dependent constants, and investigate tables indexed by
   the possible remaining count where those tables replace expensive repeated
   calculations. This preserves the multinomial estimator but requires fresh
   distributional and precision validation. Benefits are unmeasured.
3. **Budget memory across useful workspaces.** Keep the demonstrated cross-group
   batching; reserve memory for state preparation/sampler tables rather than
   maximizing the number of simultaneously retained distributions. Bound cache
   lifetimes and workspace sizes by the whole pipeline's peak memory. The
   prototype graph cache is an experiment, not a bounded production cache.

For scale, halving the measured 14.5-second bootstrap stage would reduce a
24.5-second call to about 17.3 seconds if other stages stayed unchanged. That
is an arithmetic scenario, not a forecast. Eliminating compression entirely
would save at most its approximately 4.5 seconds. These estimates help choose
where implementation work can matter.

## Reproduce

```bash
conda activate torch
python experimental/gpu_acceleration/memory_speed_bench.py
python experimental/gpu_acceleration/memory_speed_bench.py \
  --modes wide graph_exact graph_exact wide \
  --out experimental/gpu_acceleration/results_memory_exact
python experimental/gpu_acceleration/profile_sampler.py
```

For a smaller correctness run (its validation work is included in wall time):

```bash
python experimental/gpu_acceleration/memory_speed_bench.py \
  --genes 32 --boots 1000 --batch 32 --row-batch 256 \
  --modes baseline wide graph --validate \
  --out experimental/gpu_acceleration/results_memory_smoke
```
