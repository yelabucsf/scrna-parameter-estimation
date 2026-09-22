"""Memory-for-speed experiments: batch gene-group pairs and cache CUDA graphs.

No package backend changes. Graphs retain private buffers and use the default
graph-aware torch CUDA RNG. Each replay draws fresh binomials.
"""
import time
import numpy as np
import torch
from memento import bootstrap
from gpu_1d import GPUDispatch, fill_positive, regress, sync


def pack_states(states, qs):
    g = len(states)
    k = max(len(s[3]) for s in states)
    packed = np.zeros((3, k, g), dtype=np.float32)
    ns = np.empty(g, dtype=np.float32)
    for i, ((inv, inv2, expr, counts), q) in enumerate(zip(states, qs)):
        n = int(counts.sum())
        if n > 2**24:
            raise ValueError('float32 counts require n <= 2**24')
        ns[i] = n
        remaining = np.cumsum(counts[::-1], dtype=np.int64)[::-1]
        e = expr[:, 0].astype(np.float64)
        packed[0, :len(counts), i] = np.clip(counts.astype(np.float64)/remaining, 0, 1)
        packed[1, :len(counts), i] = e * inv[:, 0]
        packed[2, :len(counts), i] = (e**2-(1-q)*e) * inv2[:, 0]
    return packed, ns


def sample_packed(coeff, ns_t, boots, return_weights=False):
    """Unchanged eager binomial loop, accepting already resident coefficients."""
    g = ns_t.shape[0]
    rem = ns_t.expand(g, boots).clone()
    m1 = torch.zeros_like(rem)
    m2 = torch.zeros_like(rem)
    weights = []
    for j in range(coeff.shape[1]):
        d = torch.binomial(rem, coeff[0, j, :, None].expand_as(rem))
        if return_weights:
            weights.append(d)
        rem.sub_(d)
        m1.addcmul_(coeff[1, j, :, None], d)
        m2.addcmul_(coeff[2, j, :, None], d)
    mean = m1/ns_t
    result = (mean, m2/ns_t-mean.square())
    return result, rem, weights


class GraphBlock:
    """Replay a fixed block of state updates on mutable, persistent inputs."""
    def __init__(self, rows, boots, block=64):
        self.coeff = torch.zeros((3, block, rows), device='cuda')
        self.rem = torch.full((rows, boots), 100., device='cuda')
        self.m1 = torch.zeros_like(self.rem)
        self.m2 = torch.zeros_like(self.rem)
        self.block = block
        self.coeff[0].fill_(.01)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            self.step()
        torch.cuda.current_stream().wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.step()

    def step(self):
        for j in range(self.block):
            d = torch.binomial(self.rem, self.coeff[0, j, :, None].expand_as(self.rem))
            self.rem.sub_(d)
            self.m1.addcmul_(self.coeff[1, j, :, None], d)
            self.m2.addcmul_(self.coeff[2, j, :, None], d)


class MixedSampler:
    def __init__(self, graphs=False, exact=False):
        self.graphs = graphs
        self.exact = exact
        self.cache = {}
        self.capture_seconds = 0.
        self.calls = 0
        self.replays = 0
        self.last_remaining = None

    def __call__(self, states, qs, boots, return_weights=False):
        packed, ns = pack_states(states, qs)
        g, k = len(states), packed.shape[1]
        ns_t = torch.as_tensor(ns, device='cuda')[:, None]
        self.calls += 1
        if not self.graphs:
            coeff = torch.as_tensor(packed, device='cuda')
            result, rem, weights = sample_packed(coeff,ns_t,boots,return_weights)
            self.last_remaining = rem
            return (*result, torch.stack(weights)) if return_weights else result
        if return_weights:
            raise ValueError('Diagnostic weights are available only in eager mode')
        rows = g if self.exact else 1 << (g-1).bit_length()
        block_states = k if self.exact else 64
        key = rows, boots, block_states
        if key not in self.cache:
            sync(); start = time.perf_counter()
            self.cache[key] = GraphBlock(rows, boots, block_states)
            sync(); self.capture_seconds += time.perf_counter()-start
        block = self.cache[key]
        block.rem.zero_()
        block.rem[:g].copy_(ns_t.expand(g, boots))
        block.m1.zero_(); block.m2.zero_()
        # Prepack all blocks once, including row and trailing-state padding.
        nblocks = (k+block.block-1)//block.block
        padded = np.zeros((3, nblocks*block.block, rows), dtype=np.float32)
        padded[:, :k, :g] = packed
        coeff = torch.as_tensor(padded, device='cuda')
        for start in range(0, nblocks*block.block, block.block):
            block.coeff.copy_(coeff[:, start:start+block.block])
            block.graph.replay()
            self.replays += 1
        self.last_remaining = block.rem[:g]
        # Allocate independent outputs: cached graph storage will be reused.
        mean = block.m1[:g]/ns_t
        return mean, block.m2[:g]/ns_t-mean.square()


class WideGPUDispatch(GPUDispatch):
    """Group all independent gene-group pairs by state count before sampling."""
    def __init__(self, batch_genes=256, row_batch=1024, graphs=False, validate=False, graph_exact=False):
        super().__init__(batch_genes, 'gpu', validate)
        self.row_batch = row_batch
        self.sampler = MixedSampler(graphs, graph_exact)

    def make_buckets(self, tasks):
        buckets = {}
        for i,t in enumerate(tasks):
            for group in range(len(t['cells'])):
                mu, rv = t['true_mean'][group], t['true_res_var'][group]
                if np.isnan(mu) or np.isnan(rv) or mu == 0 or rv < 0:
                    continue
                state = bootstrap._unique_expr(t['cells'][group], t['approx_sf'][group])
                if len(state[3]) > 1:
                    buckets.setdefault((len(state[3])-1)//64, []).append((i,group,state))
        return buckets

    @staticmethod
    def state_count(state):
        return len(state[3])

    def sample_rows(self, states, qs, boots):
        return self.sampler(states,qs,boots)

    def run_chunk(self, tasks):
        g, h, b = len(tasks), len(tasks[0]['cells']), tasks[0]['num_boot']
        means = torch.full((g,h,b+1), torch.nan, device='cuda', dtype=torch.float64)
        variances = torch.full_like(means, torch.nan)
        good = torch.zeros((g,h), device='cuda', dtype=torch.bool)
        sync(); start = time.perf_counter()
        buckets = self.make_buckets(tasks)
        self.timings['compression'] += time.perf_counter()-start
        for entries in buckets.values():
            entries.sort(key=lambda row:self.state_count(row[2]))
            for offset in range(0,len(entries),self.row_batch):
                rows = entries[offset:offset+self.row_batch]
                ids, groups, states = zip(*rows)
                qs = [tasks[i]['q'][group] for i,group in zip(ids,groups)]
                self.padding_real += sum(self.state_count(s) for s in states)
                self.padding_total += len(states)*max(self.state_count(s) for s in states)
                # Retain the package's existing configurable workspace limit.
                step = bootstrap._get_batch_size(2*len(states),b,None)
                sync(); start=time.perf_counter()
                mus, vs = [], []
                for lo in range(0,b,step):
                    mu, v = self.sample_rows(states,qs,min(step,b-lo))
                    mus.append(mu); vs.append(v)
                mu, v = torch.cat(mus,-1).double(), torch.cat(vs,-1).double()
                sync(); self.timings['bootstrap'] += time.perf_counter()-start
                start=time.perf_counter()
                fit = torch.as_tensor(np.array([tasks[i]['mv_fit'][group] for i,group in zip(ids,groups)]),
                                      device='cuda',dtype=torch.float64)
                lm = mu.log()
                rv = torch.exp(v.log()-(fit[:,0,None]*lm+fit[:,1,None])*lm-fit[:,2,None])
                rv = torch.where((mu>0)&(v>0),rv,torch.nan)
                mu, valid_m = fill_positive(mu)
                rv, valid_v = fill_positive(rv)
                ii = torch.as_tensor(ids,device='cuda')
                jj = torch.as_tensor(groups,device='cuda')
                means[ii,jj,1:] = mu.log()
                variances[ii,jj,1:] = rv.log()
                means[ii,jj,0] = torch.as_tensor(np.log([tasks[i]['true_mean'][j] for i,j in zip(ids,groups)]),device='cuda')
                variances[ii,jj,0] = torch.as_tensor(np.log([tasks[i]['true_res_var'][j] for i,j in zip(ids,groups)]),device='cuda')
                good[ii,jj] = valid_m & valid_v
                sync(); self.timings['transform'] += time.perf_counter()-start
        sync(); start=time.perf_counter()
        result = regress(means,variances,good,tasks)
        sync(); self.timings['regression'] += time.perf_counter()-start
        if self.validate:
            cpu=np.asarray(self.cpu_regress(means,variances,good,tasks))
            np.testing.assert_allclose(result,cpu,rtol=2e-7,atol=2e-10,equal_nan=True)
            self.checks.append(float(np.nanmax(abs(np.asarray(result)-cpu))))
        return result
