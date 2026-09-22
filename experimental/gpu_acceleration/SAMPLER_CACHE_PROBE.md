# Sampler cache feasibility probe

Follow-up: the [standalone CUDA benchmark](SAMPLER_CACHE_RESULTS.md) is complete
and finds no useful speedup from probability-only caching. This document records
the preceding feasibility probe, not a performance claim.

The input reuse and memory estimates justify a small caching microbenchmark.
They do not demonstrate a speedup. Start with probability-only constants before
building tables for the larger-mean rejection sampler.

## What was measured

`probe_sampler_cache.py` inventories all eligible gene/group states from the
prepared CD14+ Monocyte dataset: 1,742 genes, 16 groups, 112–778 cells per group.
It uses the same true-moment eligibility checks and state order as the GPU
dispatcher. Of 27,872 possible gene/group rows, 26,987 are eligible for sampling.

For observed inputs, it runs the existing `torch.binomial` conditional chain on
64 expression-stratified genes with 10,000 replicates, then confirms with 128
genes and another seed. These yield 986 and 1,978 eligible gene/group rows.
Remaining counts are histogrammed immediately before each binomial draw. No
custom sampler, moment accumulation, or regression is introduced. Padding is
excluded from the statistics. Count conservation and histogram totals are checked.

Both runs use torch 2.9.0+cu128 and the RTX 3060, with CPU affinity restricted to
cores 0 and 1 and numerical-library threads restricted to one. Instrumented runs
take about 10 and 18 seconds, respectively; these are not performance benchmarks.

## Which computations would caching target?

The installed PyTorch header and the [versioned source](https://github.com/pytorch/pytorch/blob/v2.9.0/aten/src/ATen/native/Distributions.h)
select the branch using the remaining count times `min(p, 1-p)`:

- Below 10, the small-mean method computes `log1p(-p)` once per binomial draw,
  then generates geometric waiting times. Caching can remove the first logarithm,
  but not the random draws and logarithms inside that loop.
- At or above 10, BTRS prepares constants involving the count and probability,
  followed by rejection sampling. Tables can replace setup calculations; random
  generation and candidate-dependent acceptance work remain.
- Zero remaining count or boundary probabilities return deterministically.

Probabilities are folded using the same fp32 complement operation for this
inventory. Equal probabilities mean exact equality of those fp32 values; no
approximate probability merging is proposed.

| Fraction of real-state binomial draws | 64 genes, seed 421 | 128 genes, seed 422 |
|---|---:|---:|
| Small-mean branch | 81.53% | 81.04% |
| BTRS | 16.53% | 16.97% |
| Deterministic | 1.94% | 1.99% |

These are draw-weighted proportions, **not time proportions**. BTRS draws may
cost more. The two runs vary both gene selection and seed, so they provide a
robustness check rather than an isolated estimate of seed variation.

## Memory and reuse

The full inventory has 1,466,122 nontrivial state positions, but only **9,422
distinct folded probabilities**. A table can share constants across genes and
groups when its probability and remaining-count inputs match.

| Hypothetical allocation over the full dataset | Size |
|---|---:|
| Two fp64 constants per distinct probability | 147 KiB |
| Two fp64 constants per state, without deduplication | 22.4 MiB |
| Eight fp64 constants for all remaining counts, separately per state | 32.2 GiB |
| Same count table, shared by exact probability | 410 MiB |
| Shared table restricted to BTRS count/probability combinations | 184 MiB |

Eight fp64 fields is a conservative budgeting assumption, not an implemented
layout or precision decision. fp32 storage would halve these field allocations.
Sizes exclude mapping indices, alignment and construction temporaries. Dense
tables use counts from zero through the largest applicable group size; entries
restricted to BTRS satisfy its branch condition. Construction and lookup costs
remain unmeasured.

In both observed runs, the median state visits **51 positive remaining counts**
across 10,000 replicates. The median number of entries covering 90% and 99% of
positive-count draws is 23 and 36, respectively. An observed state/count entry
is used a median of 95 times. Thus reuse exists even without sharing across genes.

Observed support is not complete mathematical support. A sparse cache based on
these observations must fall back to direct calculation on a miss; dropping rare
outcomes would change the bootstrap distribution.

## Decision and next bounded experiment

Proceed to a standalone binomial microbenchmark of direct calculation versus
precomputed `log1p(-p)` first. Keep RNG, launch layout, arithmetic precision and
the rest of the algorithm identical. Test both a value supplied per state and
a shared lookup table: the latter saves memory but adds indexing. Include table
construction and use the observed input mixture rather than one repeated pair.

If probability caching helps, test BTRS setup tables separately. Their footprint
fits comfortably in 12 GB when shared, but memory feasibility alone does not
justify their complexity. Neither cache removes the rejection loop itself.

The local NPZ artifacts preserve the empirical distribution of folded probability,
remaining count and frequency for future microbenchmarks. They do not preserve
launch order or within-warp correlations; an integration decision needs realistic
layout tests and ultimately full-call timings. No speedup or distributional
validation of a new sampler is claimed here.

## Reproduce

```bash
conda activate torch
python experimental/gpu_acceleration/probe_sampler_cache.py
python experimental/gpu_acceleration/probe_sampler_cache.py \
  --genes 128 --seed 422 \
  --out experimental/gpu_acceleration/results_sampler_cache_confirm
```

Reports: [64 genes](results_sampler_cache/report.json),
[128 genes](results_sampler_cache_confirm/report.json).
