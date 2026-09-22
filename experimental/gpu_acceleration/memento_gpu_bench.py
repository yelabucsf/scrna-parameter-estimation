"""
memento_gpu_bench.py -- single-core CPU vs GPU, one script.

    python memento_gpu_bench.py

Measures the memento bootstrap inner loop (multinomial resampling + the
_hyper_1d_relative moment estimator) on one CPU core and on the GPU, at the
(cells, unique-states) shapes taken from a real CPU profile, and reports the
speedup per shape.

CPU arm mirrors memento's current code path: rng.multinomial() to materialize
a (k, num_boot) weight matrix, then reduce over the state axis.

GPU arm uses the exact conditional-binomial chain, FUSED with the moment
accumulation: it draws one state's weights at a time and folds them straight
into the running sums, so the (k, num_boot) matrix is never allocated. Peak
VRAM is O(genes * num_boot), not O(genes * k * num_boot) -- ~100x less, which
is what makes a 12 GB card able to batch hundreds of genes at once.

Both arms compute the same estimator:
    M1  = sum_j expr_j * W_j * inv_sf_j       / n
    M2  = sum_j (expr_j^2 - (1-q)*expr_j) * W_j * inv_sf_j^2 / n
    var = M2 - M1^2
"""
import os

# single core, honestly -- set before numpy import or BLAS will grab threads
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import time
import numpy as np

NUM_BOOT = 10_000
Q = 0.07

# (cells, unique states) -- from profiling real memento runs
SHAPES = [
    (200, 66, "small group, low expr"),
    (1000, 270, "mid group, high expr"),
    (5000, 359, "large group, high expr"),
    (20000, 526, "pooled, high expr"),
]


def make_case(n_cells, k, seed=0):
    """Per-state probabilities, expression values and inverse size factors."""
    rng = np.random.default_rng(seed)
    p = rng.dirichlet(np.ones(k) * 0.3)
    expr = rng.integers(0, 8, size=k).astype(np.float64)
    inv_sf = rng.random(k) + 0.5
    return p, expr, inv_sf


def timed(fn, reps, warmup=1, cuda=False):
    sync = None
    if cuda:
        import torch
        sync = torch.cuda.synchronize
    for _ in range(warmup):
        fn()
    if sync:
        sync()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    if sync:
        sync()
    return (time.perf_counter() - t0) / reps


# --------------------------------------------------------------------- CPU arm
def cpu_gene(p, expr, inv_sf, n_cells, num_boot, rng):
    """One gene-group, the way memento does it now."""
    W = rng.multinomial(n_cells, p, size=num_boot).T          # (k, num_boot)
    e = expr[:, None]
    i1 = inv_sf[:, None]
    i2 = i1 ** 2
    m1 = (e * W * i1).sum(0) / n_cells
    m2 = (e ** 2 * W * i2 - (1 - Q) * e * W * i2).sum(0) / n_cells
    return m1, m2 - m1 ** 2


# --------------------------------------------------------------------- GPU arm
def gpu_batch(p, expr, inv_sf, n_cells, num_boot, genes, dev, torch):
    """`genes` gene-groups at once, fused sampling + moment accumulation.

    The conditional probability p_j / (1 - sum_{i<j} p_i) is deterministic,
    so it is precomputed once in fp64. Only the remaining count is stochastic.
    """
    k = p.shape[0]
    pt = torch.as_tensor(p, device=dev, dtype=torch.float64)
    before = torch.cat([torch.zeros(1, device=dev, dtype=torch.float64),
                        torch.cumsum(pt, 0)[:-1]])
    cond = torch.clamp(pt / torch.clamp(1.0 - before, min=1e-300), 0.0, 1.0)
    cond = cond.float()

    e = torch.as_tensor(expr, device=dev, dtype=torch.float32)
    i1 = torch.as_tensor(inv_sf, device=dev, dtype=torch.float32)
    i2 = i1 ** 2
    c1 = e * i1                                   # coefficient for M1
    c2 = e ** 2 * i2 - (1 - Q) * e * i2           # coefficient for M2

    B = genes * num_boot
    m1 = torch.zeros(B, device=dev, dtype=torch.float32)
    m2 = torch.zeros(B, device=dev, dtype=torch.float32)
    rem = torch.full((B,), float(n_cells), device=dev, dtype=torch.float32)

    for j in range(k - 1):
        d = torch.binomial(rem, cond[j].expand(B))
        rem -= d
        m1 += c1[j] * d
        m2 += c2[j] * d
    m1 += c1[k - 1] * rem                         # last state takes the rest
    m2 += c2[k - 1] * rem
    m1 /= n_cells
    m2 /= n_cells
    return m1, m2 - m1 ** 2


def main():
    print("=" * 74)
    print("memento bootstrap: single-core CPU vs GPU")
    print("=" * 74)

    torch, has_gpu, why = None, False, ""
    try:
        import torch
        has_gpu = torch.cuda.is_available()
        if not has_gpu:
            why = "torch installed but no CUDA device visible"
    except Exception as exc:                 # broken install, missing libs, etc
        why = f"{type(exc).__name__}: {exc}"[:110]

    if has_gpu:
        dev = torch.device("cuda")
        free, total = torch.cuda.mem_get_info()
        print(f"GPU : {torch.cuda.get_device_name(0)}  "
              f"({total/1e9:.1f} GB, {free/1e9:.1f} free)")
        print(f"torch {torch.__version__}, CUDA {torch.version.cuda}")
    else:
        dev = None
        print("GPU : NOT AVAILABLE -- CPU numbers only")
        print(f"      ({why})")
    print(f"CPU : {os.cpu_count()} cores present, pinned to 1 for this test")
    print(f"num_boot = {NUM_BOOT}\n")

    rng = np.random.default_rng(0)
    rows = []

    print(f"{'cells':>6} {'states':>7} {'genes':>6} | {'CPU 1-core':>12} "
          f"{'GPU':>10} {'speedup':>9} | description")
    print("-" * 74)

    for n_cells, k, label in SHAPES:
        p, expr, inv_sf = make_case(n_cells, k)

        reps = 2 if n_cells >= 5000 else 3
        t_cpu = timed(lambda: cpu_gene(p, expr, inv_sf, n_cells, NUM_BOOT, rng),
                      reps=reps)

        if not has_gpu:
            print(f"{n_cells:>6} {k:>7} {'--':>6} | {t_cpu*1e3:>11.1f}m "
                  f"{'--':>10} {'--':>9} | {label}")
            rows.append((n_cells, k, t_cpu, None, None))
            continue

        # scale the gene batch to the card; fused design keeps this cheap
        best = None
        for genes in (16, 64, 256):
            try:
                t = timed(lambda: gpu_batch(p, expr, inv_sf, n_cells,
                                            NUM_BOOT, genes, dev, torch),
                          reps=2, cuda=True)
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    torch.cuda.empty_cache()
                    break
                raise
            per_gene = t / genes
            if best is None or per_gene < best[1]:
                best = (genes, per_gene)
        genes, t_gpu = best
        rows.append((n_cells, k, t_cpu, t_gpu, genes))
        print(f"{n_cells:>6} {k:>7} {genes:>6} | {t_cpu*1e3:>11.1f}m "
              f"{t_gpu*1e3:>9.2f}m {t_cpu/t_gpu:>8.1f}x | {label}")

    if not has_gpu:
        print("\nInstall torch with CUDA and rerun for the comparison.")
        return

    # sanity: GPU must reproduce the CPU estimator, not just run fast
    print("\nsanity check (GPU vs CPU estimator, 1000 cells / 270 states)")
    p, expr, inv_sf = make_case(1000, 270)
    m1_c, v_c = cpu_gene(p, expr, inv_sf, 1000, 40000, np.random.default_rng(1))
    m1_g, v_g = gpu_batch(p, expr, inv_sf, 1000, 40000, 1, dev, torch)
    m1_g = m1_g.double().cpu().numpy()
    v_g = v_g.double().cpu().numpy()
    print(f"  mean : CPU {m1_c.mean():.5f} / GPU {m1_g.mean():.5f}   "
          f"bootstrap SD ratio {m1_g.std()/m1_c.std():.4f}")
    print(f"  var  : CPU {v_c.mean():.5f} / GPU {v_g.mean():.5f}   "
          f"bootstrap SD ratio {v_g.std()/v_c.std():.4f}")
    print("  (ratios should be ~1.00; ~0.7% Monte Carlo noise at 40k boots)")

    # extrapolate to a real run
    ok = [r for r in rows if r[3]]
    speedups = [r[2] / r[3] for r in ok]
    lo, hi = min(speedups), max(speedups)
    n_cores = os.cpu_count() or 1
    print("\n" + "=" * 74)
    print(f"per gene-group speedup vs 1 core : {lo:.1f}x - {hi:.1f}x")
    print(f"vs all {n_cores} cores (your real baseline) : "
          f"{lo/n_cores:.1f}x - {hi/n_cores:.1f}x")
    mid = float(np.median(speedups)) / n_cores
    hours = 5000 * 50 * float(np.median([r[2] for r in ok])) / 3600 / n_cores
    print(f"\n5000 genes x 50 groups, {n_cores} cores : ~{hours:.1f} h")
    print(f"                         same on GPU : ~{hours/mid*60:.0f} min "
          f"(at {mid:.1f}x)")
    print("=" * 74)


if __name__ == "__main__":
    main()
