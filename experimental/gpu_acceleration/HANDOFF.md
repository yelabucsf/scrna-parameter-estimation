# memento GPU bootstrap — handoff

## Goal

Add an optional GPU path for the bootstrap inner loop in
`yelabucsf/scrna-parameter-estimation` (`memento/bootstrap.py`,
`memento/estimator.py`). Target hardware: **consumer GPUs** (RTX 3060 12 GB),
so labs without cluster access can run memento on a workstation. This is an
accessibility goal, not a cluster-performance goal.

The CPU path stays the default and the reference implementation.

## Measured: CPU profile

Per gene-group, `num_boot=10000`, single core, `_hyper_1d_relative`:

| cells | states (k) | `_unique_expr` | sampling | moments | total |
|---|---|---|---|---|---|
| 200 | 66 | 1.1% | 57% | 42% | 116 ms |
| 1,000 | 270 | 0.2% | 56% | 44% | 432 ms |
| 5,000 | 359 | 0.6% | 60% | 40% | 739 ms |
| 20,000 | 526 | 0.2% | 58% | 42% | 993 ms |

Sampling cost is **~90–105 ns per binomial draw, linear in k** — numpy's
multinomial uses the conditional-binomial method, so each replicate costs k
binomial draws. This rate was reproducible across two different machines.

## Measured: GPU result (RTX 3060 12 GB, torch 2.9.0+cu128)

| cells | k | genes/batch | CPU 1-core | GPU | speedup |
|---|---|---|---|---|---|
| 200 | 66 | 256 | 77.9 ms | 0.41 ms | 188× |
| 1,000 | 270 | 256 | 352 ms | 1.89 ms | 186× |
| 5,000 | 359 | 256 | 547 ms | 5.22 ms | 105× |
| 20,000 | 526 | 256 | 847 ms | 10.2 ms | 83× |

Correctness verified: bootstrap SD ratio GPU/CPU = 0.992 (mean), 0.995 (var),
within the 0.7% Monte Carlo noise floor at 40k boots.

**Realistic end-to-end expectation: 10–25×**, not 83–188×. See Pitfalls.

## The design: fused conditional binomial

Draw one state's weights at a time and fold them straight into the running
moment sums. Never materialize the `(k, num_boot)` weight matrix.

Key insight: the conditional probability `p_j / (1 - Σ_{i<j} p_i)` is
**deterministic**, so precompute it once in fp64. Only the remaining *count*
is stochastic. This eliminates fp32 drift concerns in the probability chain.

```python
# precompute, fp64, once per gene
before = cat([zeros(1), cumsum(p)[:-1]])
cond   = clamp(p / clamp(1 - before, min=1e-300), 0, 1).float()
c1 = expr * inv_sf                              # M1 coefficient
c2 = expr**2 * inv_sf**2 - (1-q) * expr * inv_sf**2   # M2 coefficient

# fused loop, fp32
m1 = zeros(B); m2 = zeros(B); rem = full(B, n_cells)
for j in range(k - 1):
    d = torch.binomial(rem, cond[j].expand(B))
    rem -= d
    m1 += c1[j] * d
    m2 += c2[j] * d
m1 += c1[k-1] * rem; m2 += c2[k-1] * rem        # last state takes remainder
m1 /= n_cells; m2 /= n_cells
var = m2 - m1**2
```

Peak VRAM is `O(genes × num_boot)` not `O(genes × k × num_boot)` — ~30 MB at
256 genes instead of ~3 GB. This is what makes a 12 GB card viable.

- **fp32 throughout** including counts. Verified: ~3e-7 relative error on the
  variance vs ~1e-2 Monte Carlo noise. Keep the `clamp` guards.
- The k-loop is sequential but each launch is parallel over `genes × num_boot`;
  with ~2M elements per launch, launch overhead is negligible.

## Rejected approaches — do not revisit

**Poissonized bootstrap** (independent `Poisson(n·p_j)` weights instead of
multinomial). Tested and rejected: inflates bootstrap SE by **16–57%**, and the
error does *not* shrink with n (1.16× at both 200 and 5,000 cells) — it's a
specification error, not an asymptotic approximation. Cause: memento's
estimators divide by fixed `n_obs`, but Poisson weights don't sum to n, so you
bootstrap a total rather than a mean. Self-normalizing fixes it to ~1% but
changes the estimator definition. Not worth it — exact multinomial is
achievable on GPU with no meaningful cost.

(Note: `_pseudobulk` already self-normalizes via
`(bootstrap_freq/inverse_size_factor).sum(axis=0)`, so it would have survived
Poissonization while `_hyper_1d_relative`, `_poisson_1d_relative` and
`_mean_only_1p` would not. That inconsistency would have been nasty to debug.)

**CDF inversion + scatter_add.** Exact and fully parallel, but allocates
`(genes × num_boot, n_cells)` uniforms — 6.4 billion doubles at realistic
shapes. OOMs instantly on 12 GB. Only competitive for tiny groups.

## Pitfalls — these decide whether you get 10× or 25×

1. **`_unique_expr` is now the bottleneck (Amdahl).** It's 0.2–1.2% of CPU
   time, which was negligible at 10× but eats **27–67%** of the gain at 100×.
   The 188× small-group kernel is only ~62× end-to-end. Vectorize it or move
   it to GPU; this is the single highest-value optimization.

2. **Do not return bootstrap distributions over PCIe.** `2 × 256 × 10000` fp32
   is ~20 MB ≈ 1.7 ms — a 4× overhead against 0.41 ms of kernel time on small
   groups. Reduce on device to mean / SE / ASL and return a few floats per
   gene. `main.py` only consumes summary statistics.

3. **Batch size ceiling not found.** 256 was optimal at *every* shape, using
   only ~30 MB of 11.8 GB. Sweep 1024 and 4096 — small-group shapes are
   launch-overhead-bound and should keep improving.

4. **Bucket genes by state count k before batching.** k ranges ~50–500. Padding
   with zero-probability states is correct but wasteful; the benchmark used one
   repeated gene so it understates this cost. Padded states need the final
   `out[k-1] = rem` assignment handled per-gene, not globally.

5. **Speedup declines with group size** (188× → 83×) because the rejection
   sampler diverges more at large counts — per-state GPU cost triples from
   small to pooled shapes while CPU rises only 1.4×. Mostly fine: per-donor
   groups of a few hundred cells are the common case and the best case.

6. **`os.cpu_count()` may be hyperthreaded.** 12 reported = 6 physical + HT.
   Binomial sampling is compute-bound where HT buys ~20–30%, not 2×. Use ~7.5
   as the effective divisor, not 12.

## Correctness criteria

Any GPU sampler must satisfy, versus `rng.multinomial`:

- `sum(W, axis=0) == n_cells` exactly, on every draw
- per-state mean → `n·p_j`, variance → `n·p_j(1-p_j)`
- off-diagonal covariance → `-n·p_i·p_j` (this is what distinguishes
  multinomial from Poisson — it must be negative, not zero)
- end-to-end: bootstrap SD ratio GPU/CPU ≈ 1.00 for both mean and variance

If mean ratio ≈ 1.00 but variance is off → suspect fp32 in the `M2 - M1²`
cancellation. If both off by a consistent factor → suspect the conditional
probability chain.

## Scope

- Optional extra (`memento-de[gpu]`); torch must not become a hard dependency.
  Base package currently depends only on scanpy.
- Extend the existing `_get_batch_size` logic rather than inventing new VRAM
  bounds — it already has the right shape (bounds the weight array to 64 MB).
- `_bootstrap_2d` should carry cov / var_1 / var_2 accumulators in a single
  pass, since it reuses one `gene_rvs` across three estimator calls.
- **Reproducibility is explicitly a non-goal.** GPU RNG won't match the
  `PCG64(5)` / `_spawn_task_random_states` seeding. The tagged CPU version
  reproduces the paper; that's sufficient.

## Next measurement

End-to-end on real data, not synthetic shapes: one real `ht_1d_moments` call,
CPU vs GPU, with `_unique_expr` and padding waste included. That's the number
that reflects what users see.
