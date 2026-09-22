"""Optional CUDA implementation of the hyper-relative bootstrap.

Imported only for backend='gpu'. Cell weights are shared across genes. Sampling
and moments use fp32 (TF32 disabled); transforms and regressions use fp64.
"""
import operator
import numpy as np
from sklearn.linear_model import LinearRegression

try:
    import torch
except ImportError as exc:
    raise ImportError("GPU execution requires PyTorch; install memento-de[gpu] with a CUDA-enabled torch build") from exc

from .bootstrap import _get_batch_size


def _positive_int(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f'{name} must be an integer >= {minimum}')
    try:
        value = operator.index(value)
    except TypeError as exc:
        raise ValueError(f'{name} must be an integer >= {minimum}') from exc
    if value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return value


def _generator(device, seed):
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    return gen


def _weights(n, b, device, generator):
    indices = torch.randint(n, (b, n), device=device, generator=generator)
    weights = torch.zeros((b, n), device=device, dtype=torch.float32)
    weights.scatter_add_(1, indices, torch.ones((), device=device, dtype=torch.float32).expand_as(weights))
    return weights


def _coefficients(x, sf, q, device):
    x = np.asarray(x, dtype=np.float64)
    inverse = 1 / np.asarray(sf)
    squared = inverse ** 2
    first = x * inverse[:, None]
    second = (x * x - (1 - q) * x) * squared[:, None]
    return torch.as_tensor(np.concatenate((first, second), axis=1).astype(np.float32), device=device)


def _fill_valid(x, valid, generator):
    counts = valid.sum(-1)
    if bool(valid.all()):
        return x, counts > 0
    cumulative = valid.cumsum(-1)
    ranks = (torch.rand(x.shape, device=x.device, dtype=x.dtype, generator=generator) * counts[:, None]).long() + 1
    indices = torch.searchsorted(cumulative, ranks.contiguous()).clamp_max(x.shape[-1] - 1)
    return torch.where(valid, x, x.gather(-1, indices)), counts > 0


def _fill_positive(x, generator):
    return _fill_valid(x, (x > 0) & ~torch.isnan(x), generator)


def _pair_coefficients(x, sf, q, device):
    """Columns: mean1, mean2, second1, second2, cross moment."""
    n_pairs = x.shape[1] // 2
    marginal = _coefficients(x, sf, q, device)
    squared = (1 / np.asarray(sf)) ** 2
    cross = (x[:, :n_pairs].astype(float) * x[:, n_pairs:]) * squared[:, None]
    return torch.cat((marginal, torch.as_tensor(cross, device=device, dtype=torch.float32)), dim=1)


def _pair_correlations(product, n, pairs):
    m1, m2, s1, s2, cross = (v.T / n for v in product.split(pairs, dim=1))
    v1, v2 = s1 - m1.square(), s2 - m2.square()
    corr = (cross - m1 * m2) / (v1 * v2).sqrt()
    return torch.where((v1 > 0) & (v2 > 0), corr, torch.nan).double().contiguous()


def _regression_map(covariate, treatment, weights):
    residual = np.eye(len(weights))
    residual -= LinearRegression(n_jobs=1).fit(covariate, residual, weights).predict(covariate)
    a = treatment - LinearRegression(n_jobs=1).fit(covariate, treatment, weights).predict(covariate)
    a -= np.average(a, axis=0, weights=weights)
    denominator = (a * a * weights[:, None]).sum(0)
    centered = treatment - np.average(treatment, axis=0, weights=weights)
    original_scale = (centered * centered * weights[:, None]).sum(0)
    tolerance = (16 * np.finfo(float).eps * max(len(weights), covariate.shape[1] + 1)) ** 2
    identifiable = denominator > tolerance * original_scale
    mapping = (a.T * weights) @ residual / np.where(identifiable, denominator, 1)[:, None]
    mapping[~identifiable] = np.nan
    return mapping


def _summarize(mapping, ys, observed=False):
    # Bound bootstrap-by-treatment temporaries for cis-eQTL designs with
    # hundreds or thousands of SNPs per gene. Never repeat the bootstrap.
    block = min(8, mapping.shape[-1])
    return np.concatenate([
        _summarize_block(mapping[start:start + block], ys, observed)
        for start in range(0, mapping.shape[0], block)
    ], axis=-1)


def _summarize_block(mapping, ys, observed=False):
    values = []
    for y in ys:
        coef = mapping @ y
        if coef.shape[-1] < 2:
            values.extend([torch.full(coef.shape[:-1], torch.nan, device=y.device, dtype=y.dtype)] * 3)
            continue
        null = coef[..., 1:] - coef[..., :1]
        location = null.mean(-1)
        scale = null.std(-1, correction=0)
        stat = coef[..., 0].abs()
        p = .5 * torch.erfc((stat - location) / (scale * 2**.5))
        p += .5 * torch.erfc((stat + location) / (scale * 2**.5))
        p = torch.where(scale > 0, p, torch.nan)
        values.extend((coef[..., 0] if observed else coef.mean(-1), coef[..., 1:].std(-1, correction=0), p))
    return torch.stack(values, dim=-2).cpu().numpy()


def _regress(means, variances, good, covariate, treatment, nc, designs):
    """Group matching masks/designs; retain caller gene and treatment ordering."""
    correlation = variances is None
    moments = (means,) if correlation else (means, variances)
    masks = good.cpu().numpy()
    partitions = {}
    outputs = [None] * len(designs)
    for i, (tx, cov) in enumerate(designs):
        partitions.setdefault((tuple(masks[i]), tx, cov), []).append(i)
    for (mask, tx, cov), ids in partitions.items():
        mask = np.array(mask)
        if mask.sum() < 2:
            for i in ids:
                outputs[i] = np.full((3 if correlation else 6, len(tx)), np.nan)
            continue
        mapping = _regression_map(covariate.loc[:, list(cov)].values[mask].astype(float),
                                  treatment.loc[:, list(tx)].values[mask].astype(float), nc[mask])
        mapping = torch.as_tensor(mapping, device=means.device, dtype=torch.float64)
        ix = torch.as_tensor(ids, device=means.device)
        use = torch.as_tensor(mask, device=means.device)
        ys = [y[ix][:, use] for y in moments]
        finite = torch.stack([torch.isfinite(y).all(1) for y in ys]).all(0)
        if bool(finite.all()):
            for i, result in zip(ids, _summarize(mapping, ys, observed=correlation)):
                outputs[i] = result
        else:
            for row, i in enumerate(ids):
                outputs[i] = _summarize(mapping, [y[row, :, finite[row]] for y in ys], observed=correlation)
    return outputs


def _memory_plan(nc, boots, genes, budget, batch_size, correlation=False):
    """Conservative working-memory target, excluding CUDA context/allocator cache."""
    weight_bytes = int(nc.sum()) * boots * 4
    cache = weight_bytes <= budget // 3
    retained = weight_bytes if cache else 0
    # Logs, regression copies, transform temporaries and dense coefficients.
    per_gene = (boots + 1) * (64 * len(nc) + 64) + int(nc.max()) * (120 if correlation else 24)
    available = budget - retained - budget // 8
    if available < per_gene or budget // 8 < int(nc.max()) * 12:
        raise ValueError('gpu_memory_budget is too small for one gene; increase it or reduce num_boot')
    batch = min(genes, batch_size, available // per_gene)
    chunk = min(512, boots, _get_batch_size((12 * int(nc.max()) + 7) // 8, boots, None),
                (budget // 8) // (12 * int(nc.max())))
    return int(batch), int(chunk), cache, retained


def _working_budget(free, requested):
    """Bytes available to this call; reserve headroom for other GPU users."""
    if requested is None:
        return min(int(free * .5), 8 * 1024**3)
    return min(requested, int(free * .75))


def _ht(adata, genes, indices, treatment, covariate, treatment_for_gene, covariate_for_gene,
          num_boot, random_state, memory_budget, batch_size, device, pair_positions=None, **kwargs):
    correlation = pair_positions is not None
    if kwargs.get('resample_rep', False):
        raise NotImplementedError("GPU testing currently requires resample_rep=False")
    if kwargs.get('approx', 'norm') != 'norm':
        raise NotImplementedError("GPU testing currently supports approx='norm' only")
    extra = set(kwargs) - {'resample_rep', 'approx'}
    if extra:
        raise TypeError(f'Unsupported GPU options: {sorted(extra)}')
    u = adata.uns['memento']
    if u['estimator_type'] != 'hyper_relative':
        raise NotImplementedError("GPU testing currently supports estimator_type='hyper_relative' only")
    boots = _positive_int(num_boot, 'num_boot', 2)
    batch_size = _positive_int(batch_size, 'gpu_batch_size')
    budget = None if memory_budget is None else _positive_int(memory_budget, 'gpu_memory_budget') * 1024**2
    device = torch.device(device)
    if device.type != 'cuda':
        raise ValueError('gpu_device must be a CUDA device')
    if not genes or treatment.shape[1] == 0:
        return [np.empty((3 if correlation else 6, 0)) for _ in genes], {'device': str(device), 'sampling': 'shared_cell_multinomial', 'empty': True}
    if not torch.cuda.is_available():
        raise RuntimeError('GPU execution requires an available CUDA device and a CUDA-enabled PyTorch build')
    groups = u['groups']
    if len(groups) < 2:
        raise ValueError('GPU testing requires at least two groups')
    if treatment.shape[0] != len(groups) or covariate.shape[0] != len(groups):
        raise ValueError('Treatment and covariate rows must match the memento group order')
    if not treatment.columns.is_unique or not covariate.columns.is_unique:
        raise ValueError('Treatment and covariate columns must be unique')
    if not np.isfinite(treatment.values.astype(float)).all() or not np.isfinite(covariate.values.astype(float)).all():
        raise ValueError('Treatment and covariates must be finite numeric values')
    designs = []
    for gene in genes:
        tx = tuple(treatment.columns if treatment_for_gene is None else treatment_for_gene[gene])
        cov = tuple(covariate.columns if covariate_for_gene is None else covariate_for_gene[gene])
        if not tx or not cov or not set(tx).issubset(treatment.columns) or not set(cov).issubset(covariate.columns):
            raise ValueError(f'Invalid treatment or covariate selection for gene {gene}')
        designs.append((tx, cov))
    nc = np.array([u['group_cells'][group].shape[0] for group in groups])
    if np.any(nc <= 0) or np.any(nc > 2**24):
        raise ValueError('GPU cell bootstrap requires 0 < cells per group <= 2**24')
    for group, n in zip(groups, nc):
        sf = np.asarray(u['approx_size_factor'][group])
        data = u['group_cells'][group].data
        q = u['group_q'][group]
        if sf.shape != (n,) or not np.isfinite(sf).all() or np.any(sf <= 0):
            raise ValueError('Size factors must be finite, positive and match the group cells')
        if not np.isfinite(data).all() or np.any(data < 0) or np.any(data != np.rint(data)):
            raise ValueError('GPU cell bootstrap requires finite nonnegative integer expression counts')
        if not np.isfinite(q) or not 0 <= q < 1:
            raise ValueError('Capture rates must lie in [0, 1)')
    with torch.cuda.device(device):
        free, _ = torch.cuda.mem_get_info()
        budget = _working_budget(free, budget)
        batch, chunk, cache, retained = _memory_plan(nc, boots, len(genes), budget, batch_size, correlation)
        seeds = np.random.SeedSequence(random_state).spawn(len(groups) + 1)
        seed_values = [int(s.generate_state(1, dtype=np.uint64)[0]) for s in seeds]
        fill_generator = _generator(device, seed_values[-1])
        weights = {}
        previous_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        outputs = []
        try:
            with torch.no_grad():
                if cache:
                    for gi, n in enumerate(nc):
                        gen = _generator(device, seed_values[gi])
                        w = torch.empty((boots, int(n)), device=device, dtype=torch.float32)
                        for lo in range(0, boots, chunk):
                            w[lo:lo + chunk] = _weights(int(n), min(chunk, boots - lo), device, gen)
                        weights[gi] = w
                for start in range(0, len(genes), batch):
                    take = indices[start:start + batch]
                    g = len(take)
                    means = torch.full((g, len(groups), boots + 1), torch.nan, device=device, dtype=torch.float64)
                    variances = None if correlation else torch.full_like(means, torch.nan)
                    good = torch.zeros((g, len(groups)), device=device, dtype=torch.bool)
                    for gi, (group, n) in enumerate(zip(groups, nc)):
                        columns = np.concatenate((take[:, 0], take[:, 1])) if correlation else take
                        x = u['group_cells'][group][:, columns].toarray()
                        sf = np.asarray(u['approx_size_factor'][group])
                        coef = (_pair_coefficients if correlation else _coefficients)(x, sf, u['group_q'][group], device)
                        if cache:
                            product = weights[gi] @ coef
                        else:
                            # Reset the group stream for every gene batch: streamed
                            # weights still describe the SAME cell resampling.
                            gen = _generator(device, seed_values[gi])
                            product = torch.empty((boots, coef.shape[1]), device=device, dtype=torch.float32)
                            for lo in range(0, boots, chunk):
                                w = _weights(int(n), min(chunk, boots - lo), device, gen)
                                product[lo:lo + chunk] = w @ coef
                        if correlation:
                            corr = _pair_correlations(product, int(n), g)
                            corr, valid = _fill_valid(corr, torch.isfinite(corr) & (corr.abs() < 1), fill_generator)
                            true = np.asarray(u['2d_moments'][group]['corr'])[pair_positions[start:start + g]]
                            eligible = np.isfinite(true) & (np.abs(true) < 1) & (take[:, 0] != take[:, 1])
                            means[:, gi, 1:] = corr
                            means[:, gi, 0] = torch.as_tensor(true, device=device)
                            good[:, gi] = torch.as_tensor(eligible, device=device) & valid
                            continue
                        mu32 = product[:, :g].T / int(n)
                        var32 = product[:, g:].T / int(n) - mu32.square()
                        mu = mu32.double().contiguous()
                        var = var32.double().contiguous()
                        fit = torch.as_tensor(u['mv_regressor'][group], device=device, dtype=torch.float64)
                        lm = mu.log()
                        rv = torch.exp(var.log() - (fit[0] * lm + fit[1]) * lm - fit[2])
                        rv = torch.where((mu > 0) & (var > 0), rv, torch.nan)
                        mu, valid_mean = _fill_positive(mu, fill_generator)
                        rv, valid_var = _fill_positive(rv, fill_generator)
                        true_mean = np.asarray(u['1d_moments'][group][0])[take]
                        true_var = np.asarray(u['1d_moments'][group][2])[take]
                        eligible = ~(np.isnan(true_mean) | np.isnan(true_var) | (true_mean == 0) | (true_var < 0))
                        if np.all(sf == sf[0]):
                            eligible &= np.any(x != x[:1], axis=0)
                        means[:, gi, 1:] = mu.log()
                        variances[:, gi, 1:] = rv.log()
                        with np.errstate(divide='ignore', invalid='ignore'):
                            means[:, gi, 0] = torch.as_tensor(np.log(true_mean), device=device)
                            variances[:, gi, 0] = torch.as_tensor(np.log(true_var), device=device)
                        good[:, gi] = torch.as_tensor(eligible, device=device) & valid_mean & valid_var
                    outputs.extend(_regress(means, variances, good, covariate, treatment, nc, designs[start:start + g]))
        finally:
            weights.clear()
            torch.backends.cuda.matmul.allow_tf32 = previous_tf32
    return outputs, {'device': str(device), 'working_memory_bytes': budget, 'memory_policy': 'auto' if memory_budget is None else 'explicit',
                     'gene_batch_size': batch,
                     'bootstrap_chunk_size': chunk, 'cached_cell_weights': cache, 'weight_cache_bytes': retained,
                     'sampling': 'shared_cell_multinomial', 'moment_dtype': 'float32', 'regression_dtype': 'float64',
                     'approx': 'norm', 'resample_rep': False}


def ht_1d(*args, **kwargs):
    return _ht(*args, **kwargs)


def ht_2d(adata, treatment, covariate, treatment_for_gene, covariate_for_gene,
          num_boot, random_state, memory_budget, batch_size, device, **kwargs):
    moments = adata.uns['memento']['2d_moments']
    positions = np.array([i for i, pair in enumerate(moments['gene_pairs'])
                          if treatment_for_gene is None or pair in treatment_for_gene], dtype=int)
    pairs = [moments['gene_pairs'][i] for i in positions]
    indices = np.column_stack((moments['gene_idx_1'][positions], moments['gene_idx_2'][positions]))
    return _ht(adata, pairs, indices, treatment, covariate, treatment_for_gene, covariate_for_gene,
               num_boot, random_state, memory_budget, batch_size, device, pair_positions=positions, **kwargs)
