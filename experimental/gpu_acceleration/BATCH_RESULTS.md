# Does using more of the 12 GB card help?

Only marginally for gene batching in the current implementation. All 1,742
eligible genes fit in one batch, but that does not outperform 1,024. A separate
increase in sampler working memory also fails to show a clear gain after a
timing-drift check. These are observations on this workload, not a general
statement about GPU memory or other datasets.

A subsequent [implementation experiment](MEMORY_SPEED_RESULTS.md) batches
independent gene-group pairs across donor groups and reduces full-call time to
about 24.5 seconds. It also tests graph caches and identifies the remaining
optimization targets.

Same CD14+ dataset, donor-adjusted model, 10,000 bootstraps, GPU bootstrap and
fp64 GPU regression as [the original experiments](RESULTS.md). All runs enforce
the two-logical-CPU limit. The sweep warms all stages on a small real-data call,
clears unused CUDA allocations before each trial, and tests batch sizes in
ascending then descending order. Dataset preparation and the warmup are excluded
from public-call timing.

## Gene-batch sweep, unchanged 64 MiB sampler budget

| Genes per batch | First pass | Reverse pass | Peak live tensors | Peak allocator reservation |
|---|---:|---:|---:|---:|
| 256 | 41.53 s | 29.29 s | 1.51 GiB | 4.27 GiB |
| 512 | 35.55 s | 29.42 s | 2.91 GiB | 5.22 GiB |
| 1,024 | 28.93 s | 28.62 s | 4.82 GiB | 7.67 GiB |
| 1,742 (all genes) | 30.07 s | 29.39 s | 8.70 GiB | 11.24 GiB |

Timing drift is substantial: even 256-gene batches reach about 29 seconds on the
reverse pass. Thus the first-pass 41.5 → 28.9 second improvement cannot be
attributed entirely to batch size. In the reverse pass, 1,024 beats 256 by only
2.3%, while all genes in one batch are slightly slower than either.

The reverse-pass bootstrap stages are nearly identical: 18.01, 18.27, 17.86,
and 18.13 seconds for batch sizes 256, 512, 1,024 and 1,742. CPU state compression
remains about five seconds for every size. Regression benefits somewhat from
larger batches, but accounts for only about 1.5–2.3 seconds. Padded-state work
increases from 22.7% at batch 256 to 31.4% at batch 1,742.

The largest batch reserves most of the physical card and still provides no
clear speed benefit. Allocator reservations include reusable memory, not just
live tensors; these measurements exclude other processes and CUDA-context
memory. No tested configuration ran out of CUDA memory.

## Separate sampler-memory experiment

The current GPU path reuses the package's 64 MiB bootstrap working-array
budget. Large gene buckets therefore split draws into chunks. Merely increasing
the outer gene batch does not remove this limit. A scoped experimental override
increased that budget fourfold to 256 MiB, without changing package defaults.

| Gene batch | Sampler budget | Public-call timings |
|---|---:|---:|
| 1,024 | 64 MiB, initial sweep | 28.93, 28.62 s |
| 1,024 | 256 MiB | 25.45, 26.42 s |
| 1,742 | 256 MiB | 26.73, 26.67 s |
| 1,024 | 64 MiB, subsequent confirmation | 26.49 s |

The confirmation with the original budget is essentially as fast as the last
256 MiB run (26.49 versus 26.42 seconds). CPU preprocessing also sped up during
the experiment. Consequently, the earlier apparent roughly 10% sampler-budget
gain is not established; timing drift can account for much of it. Peak live
memory stays about 4.82 GiB at batch 1,024 and 8.70 GiB at batch 1,742 because
retained distributions and regression temporaries dominate the peak.

For this dataset, 256 remains a good memory-efficient choice; 1,024 is the
fastest observed size if several extra GiB are available. There is no reason
from these results to fill the card with the largest possible gene batch.
Further gains likely need changes to preprocessing, sampling/launch overhead,
or bucketing rather than batch size alone.

## Checks and reproduction

All 13 full-data runs produce finite values for every gene and all six summary
outputs. Mean/variability SEs on the 128-gene overlap were checked against the
existing CPU reference; median ratios remain near one. This is a distributional
check because changing batching can change GPU random streams.

```bash
conda activate torch
python experimental/gpu_acceleration/batch_sweep.py
python experimental/gpu_acceleration/batch_sweep.py \
  --batches 1024 1742 --sampler-mib 256 \
  --out experimental/gpu_acceleration/results_batch_sweep256/report.json
python experimental/gpu_acceleration/batch_sweep.py \
  --batches 1024 --repeats 1 --sampler-mib 64 \
  --out experimental/gpu_acceleration/results_batch_sweep64_confirm/report.json
```

Raw reports: [gene-batch sweep](results_batch_sweep/report.json),
[larger workspace](results_batch_sweep256/report.json),
[original-budget confirmation](results_batch_sweep64_confirm/report.json),
[SE checks](results_batch_sweep/accuracy.json).
