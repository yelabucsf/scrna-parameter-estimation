"""Experimental batched hyper-relative bootstrap and default 1D regression.

Not a public backend: supports common treatment/covariates, approx='norm',
and resample_rep=False only. CPU remains the package default.
"""
import time
from contextlib import contextmanager
from unittest.mock import patch

import numpy as np
import torch
from sklearn.linear_model import LinearRegression

from memento import bootstrap, hypothesis_test as ht
import memento.main as main


def sync():
    torch.cuda.synchronize()


def sample_states(states, n, q, num_boot, return_weights=False):
    """Variable-k rows, exact multinomial counts; last real state gets remainder.

    States are CPU _unique_expr tuples. Probability preparation is float64;
    counts and fused moment accumulation are float32. No weight cube exists.
    """
    if n > 2**24:
        raise ValueError("float32 counts require n <= 2**24")
    g = len(states)
    kmax = max(len(s[3]) for s in states)
    cond = np.zeros((kmax, g), dtype=np.float32)
    c1 = np.zeros_like(cond)
    c2 = np.zeros_like(cond)
    for i, (inv, inv2, expr, counts) in enumerate(states):
        k = len(counts)
        # Integer suffix sums avoid cancellation in 1-cumsum(p).
        remaining = np.cumsum(counts[::-1], dtype=np.int64)[::-1]
        cond[:k, i] = np.clip(counts.astype(np.float64) / remaining, 0, 1)
        e = expr[:, 0].astype(np.float64)
        c1[:k, i] = e * inv[:, 0]
        c2[:k, i] = (e**2 - (1-q)*e) * inv2[:, 0]
    cond, c1, c2 = [torch.as_tensor(x, device="cuda") for x in (cond, c1, c2)]
    m1 = torch.zeros((g, num_boot), device="cuda")
    m2 = torch.zeros_like(m1)
    rem = torch.full_like(m1, n)
    weights = [] if return_weights else None
    for j in range(kmax):
        # p=1 consumes the remainder at each row's own final real state;
        # subsequent padded rows have p=0 and coefficients=0.
        d = torch.binomial(rem, cond[j, :, None].expand_as(rem))
        if return_weights:
            weights.append(d)
        rem -= d
        m1.addcmul_(c1[j, :, None], d)
        m2.addcmul_(c2[j, :, None], d)
    m1 /= n
    m2 /= n
    result = (m1, m2 - m1.square())
    return (*result, torch.stack(weights)) if return_weights else result


def fill_positive(x):
    """Uniformly replace invalid draws from that row's valid draws, like _fill."""
    valid = (x > 0) & ~torch.isnan(x)
    counts = valid.sum(-1)
    # searchsorted of cumulative counts samples only valid positions.
    cumulative = valid.cumsum(-1)
    ranks = (torch.rand_like(x) * counts[:, None]).long() + 1
    idx = torch.searchsorted(cumulative, ranks).clamp_max(x.shape[-1]-1)
    filled = torch.where(valid, x, x.gather(-1, idx))
    return filled, counts > 0


def regression_map(covariate, treatment, weights):
    """Small CPU design factorization, reused for every bootstrap replicate.

    Weighted residualization includes sklearn's fitted intercept. Cross-coef
    estimates each treatment marginally, exactly as the reference does.
    """
    r = np.eye(len(weights))
    r -= LinearRegression(n_jobs=1).fit(covariate, r, weights).predict(covariate)
    a = treatment - LinearRegression(n_jobs=1).fit(
        covariate, treatment, weights).predict(covariate)
    a -= np.average(a, axis=0, weights=weights)
    denom = (a*a*weights[:, None]).sum(0)
    result = (a.T * weights) @ r / np.where(denom > 0, denom, 1)[:, None]
    result[denom == 0] = np.nan
    return result


def regress(log_mean, log_var, good, tasks, device="cuda", dtype=torch.float64):
    """Batched regression, with identical group and finite-draw filters to CPU.

    Only small design matrices and six summaries per gene cross PCIe for the
    resident path. Float64 protects tail probabilities on the consumer GPU.
    """
    outputs = [None] * len(tasks)
    masks = good.cpu().numpy()
    for mask in np.unique(masks, axis=0):
        ids = np.flatnonzero(np.all(masks == mask, axis=1))
        if mask.sum() < 2:
            for i in ids:
                outputs[i] = np.full((6, tasks[i]['treatment'].shape[1]), np.nan)
            continue
        task = tasks[ids[0]]
        mapping = regression_map(task['covariate'][mask], task['treatment'][mask],
                                 task['Nc_list'][mask])
        mapping = torch.as_tensor(mapping, device=device, dtype=dtype)
        ids_t = torch.as_tensor(ids, device=log_mean.device)
        mask_t = torch.as_tensor(mask, device=log_mean.device)
        ys = [y[ids_t][:, mask_t].to(device=device, dtype=dtype)
              for y in (log_mean, log_var)]
        finite = torch.isfinite(ys[0]).all(1) & torch.isfinite(ys[1]).all(1)
        def summarize(ys):
            vals = []
            for y in ys:
                coef = mapping @ y
                if coef.shape[-1] < 2:
                    vals.extend([torch.full(coef.shape[:-1], float('nan'), device=device)]*3)
                    continue
                null = coef[..., 1:] - coef[..., :1]
                loc = null.mean(-1)
                scale = null.std(-1, correction=0)
                stat = coef[..., 0].abs()
                p = .5 * torch.erfc((stat-loc)/(scale*2**.5))
                p += .5 * torch.erfc((stat+loc)/(scale*2**.5))
                p = torch.where(scale > 0, p, torch.nan)
                vals.extend((coef.mean(-1), coef[..., 1:].std(-1, correction=0), p))
            return torch.stack(vals, dim=-2).double().cpu().numpy()
        if bool(finite.all()):
            for i, out in zip(ids, summarize(ys)):
                outputs[i] = out
        else:
            # Preserve common finite-draw filtering for overflow edge cases.
            for row, i in enumerate(ids):
                outputs[i] = summarize([y[row, :, finite[row]] for y in ys])
    return outputs


class GPUDispatch:
    """A scoped replacement for main.Parallel, preserving ht_1d_moments I/O."""
    def __init__(self, batch_genes=128, regression="gpu", validate=False):
        self.batch_genes = batch_genes
        self.regression = regression
        self.validate = validate
        self.timings = dict(compression=0., bootstrap=0., transform=0., regression=0.)
        self.padding_real = self.padding_total = 0
        self.checks = []

    def __call__(self, jobs):
        jobs = list(jobs)
        tasks = []
        for fn, args, kwargs in jobs:
            if fn.func is not ht._ht_1d or args or kwargs:
                raise ValueError("GPU dispatcher accepts only default 1D moment tasks")
            t = fn.keywords
            if t.get('resample_rep', False) or t.get('approx', 'norm') != 'norm':
                raise NotImplementedError("Only norm ASL, without replicate resampling")
            if t['_estimator_1d'].__name__ != '_hyper_1d_relative':
                raise NotImplementedError("Only hyper_relative")
            tasks.append(t)
        for t in tasks:
            for key in ('treatment', 'covariate', 'Nc_list'):
                if not np.array_equal(t[key], tasks[0][key]):
                    raise NotImplementedError("Common design and groups required")
        results = []
        for start in range(0, len(tasks), self.batch_genes):
            chunk = tasks[start:start+self.batch_genes]
            results.extend(self.run_chunk(chunk))
        return results

    def run_chunk(self, tasks):
        g, h, b = len(tasks), len(tasks[0]['cells']), tasks[0]['num_boot']
        means = torch.full((g, h, b+1), torch.nan, device='cuda', dtype=torch.float64)
        variances = torch.full_like(means, torch.nan)
        good = torch.zeros((g,h), dtype=torch.bool, device='cuda')
        for group in range(h):
            sync(); t0 = time.perf_counter()
            entries = []
            for i, t in enumerate(tasks):
                mu, rv = t['true_mean'][group], t['true_res_var'][group]
                if np.isnan(mu) or np.isnan(rv) or mu == 0 or rv < 0:
                    continue
                state = bootstrap._unique_expr(t['cells'][group], t['approx_sf'][group])
                if len(state[3]) > 1:
                    entries.append((i, state))
            entries.sort(key=lambda e: len(e[1][3]))
            self.timings['compression'] += time.perf_counter()-t0
            # Narrow k buckets avoid one high-expression gene padding all rows.
            buckets = {}
            for i, state in entries:
                buckets.setdefault((len(state[3])-1)//64, []).append((i, state))
            for entries in buckets.values():
                ids, states = zip(*entries)
                n, q = tasks[0]['Nc_list'][group], tasks[0]['q'][group]
                self.padding_real += sum(len(s[3]) for s in states)
                self.padding_total += len(states)*max(len(s[3]) for s in states)
                # Reuse package's 64 MiB budget: four fp32 live sampler arrays
                # have the same size as two int64 categories per gene.
                step = bootstrap._get_batch_size(2*len(states), b, None)
                sync(); t0 = time.perf_counter()
                mus, vs = [], []
                for lo in range(0, b, step):
                    mu, v = sample_states(states, int(n), q, min(step,b-lo))
                    mus.append(mu); vs.append(v)
                mu, v = torch.cat(mus, -1).double(), torch.cat(vs, -1).double()
                sync(); self.timings['bootstrap'] += time.perf_counter()-t0
                t0 = time.perf_counter()
                fit = torch.as_tensor(np.array([tasks[i]['mv_fit'][group] for i in ids]),
                                      device='cuda', dtype=torch.float64)
                lm = mu.log()
                rv = torch.exp(v.log() - (fit[:,0,None]*lm+fit[:,1,None])*lm-fit[:,2,None])
                rv = torch.where((mu > 0) & (v > 0), rv, torch.nan)
                mu, valid_m = fill_positive(mu)
                rv, valid_v = fill_positive(rv)
                idx = torch.as_tensor(ids, device='cuda')
                means[idx, group, 1:] = mu.log()
                variances[idx, group, 1:] = rv.log()
                means[idx, group, 0] = torch.as_tensor(np.log([tasks[i]['true_mean'][group] for i in ids]), device='cuda')
                variances[idx, group, 0] = torch.as_tensor(np.log([tasks[i]['true_res_var'][group] for i in ids]), device='cuda')
                good[idx, group] = valid_m & valid_v
                sync(); self.timings['transform'] += time.perf_counter()-t0
        sync(); t0 = time.perf_counter()
        if self.regression == 'cpu':
            result = self.cpu_regress(means, variances, good, tasks)
        elif self.regression == 'cpu_map':
            result = regress(means, variances, good, tasks, device='cpu')
        else:
            result = regress(means, variances, good, tasks)
        sync(); self.timings['regression'] += time.perf_counter()-t0
        if self.validate:
            # Outside measured stages: identical inputs isolate arithmetic.
            cpu = np.asarray(self.cpu_regress(means, variances, good, tasks))
            gpu = np.asarray(regress(means, variances, good, tasks))
            np.testing.assert_allclose(gpu, cpu, rtol=2e-7, atol=2e-10, equal_nan=True)
            checks = {'max_abs_error':float(np.nanmax(np.abs(gpu-cpu))), 'timings':{}}
            for name, fn in (
                ('reference_cpu_with_transfer', lambda: self.cpu_regress(means,variances,good,tasks)),
                ('mapped_cpu_with_transfer', lambda: regress(means,variances,good,tasks,device='cpu')),
                ('mapped_gpu_fp64_resident', lambda: regress(means,variances,good,tasks)),
                ('mapped_gpu_fp32_resident', lambda: regress(means,variances,good,tasks,dtype=torch.float32)),
            ):
                fn(); sync()
                times=[]
                for _ in range(3):
                    t0=time.perf_counter(); value=np.asarray(fn()); sync()
                    times.append(time.perf_counter()-t0)
                checks['timings'][name] = {'seconds':times,
                    'max_abs_error':float(np.nanmax(np.abs(value-cpu)))}
            self.checks.append(checks)
        return result

    @staticmethod
    def cpu_regress(means, variances, good, tasks):
        means, variances, good = [x.cpu().numpy() for x in (means, variances, good)]
        return [ht._regress_1d(t['covariate'][mask], t['treatment'][mask],
                means[i,mask], variances[i,mask], t['Nc_list'][mask])
                for i, (t, mask) in enumerate(zip(tasks, good))]


@contextmanager
def gpu_execution(dispatch):
    """Use only in a single-threaded experiment, never as a global backend."""
    with patch.object(main, 'Parallel', lambda **kwargs: dispatch):
        yield
