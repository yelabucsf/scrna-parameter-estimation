"""Run the real-data ht_1d_moments experiment with at most two CPU cores.

Example (from repository root, in the torch environment):
  python experimental/gpu_acceleration/real_data_bench.py --genes 128 --boots 10000
"""
import os
import argparse
import json
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import anndata
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import memento
from gpu_1d import GPUDispatch, gpu_execution

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.backends.cuda.matmul.allow_tf32 = False


def prepare(path, cell, q):
    x = anndata.read_h5ad(path, backed='r')
    a = x[x.obs.cell == cell].to_memory()
    x.file.close()
    if not a.n_obs:
        raise ValueError(f'No cells labeled {cell}')
    a.X = sp.csr_matrix(a.X)
    a.obs['q'] = q
    memento.setup_memento(a, q_column='q')
    memento.create_groups(a, label_columns=['ind', 'stim'])
    memento.compute_1d_moments(a, min_perc_group=.7)
    groups = a.uns['memento']['groups']
    metadata = a.obs.drop_duplicates('memento_group').set_index('memento_group').loc[groups]
    treatment = pd.DataFrame({'stim': (metadata.stim == 'stim').astype(float)}, index=groups)
    covariate = pd.get_dummies(metadata.ind, drop_first=True, dtype=float)
    return a, treatment, covariate


def comparison(a, b):
    out = {}
    for key in ('mean_coef', 'mean_se', 'mean_asl', 'var_coef', 'var_se', 'var_asl'):
        x,y = np.asarray(a[key]), np.asarray(b[key])
        use = np.isfinite(x) & np.isfinite(y)
        d = np.abs(y[use]-x[use])
        metrics = {'n_finite': int(use.sum()), 'nan_mismatch': int((np.isnan(x)!=np.isnan(y)).sum()),
                   'median_abs_diff': float(np.median(d)), 'max_abs_diff': float(d.max())}
        if key.endswith('_se'):
            ratio = y[use]/x[use]
            metrics['ratio_quantiles'] = np.quantile(ratio,[0,.05,.5,.95,1]).tolist()
        if key.endswith('_coef'):
            se = np.asarray(a[key.replace('coef','se')])[use]
            metrics['abs_diff_over_se_quantiles'] = np.quantile(d/se,[.5,.95,1]).tolist()
        out[key] = metrics
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='/mnt/c/Data/memento_workspace/interferon_filtered.h5ad')
    p.add_argument('--cell', default='CD14+ Monocytes')
    p.add_argument('--q', type=float, default=.07)
    p.add_argument('--genes', type=int, default=128)
    p.add_argument('--boots', type=int, default=10000)
    p.add_argument('--batch', type=int, default=128)
    p.add_argument('--cpu-jobs', type=int, choices=[1,2], default=2)
    p.add_argument('--modes', nargs='+', default=['cpu','gpu_cpu','gpu'])
    p.add_argument('--validate', action='store_true')
    p.add_argument('--out', default='experimental/gpu_acceleration/results')
    args = p.parse_args()
    if args.genes < 1 or args.boots < 2 or args.batch < 1:
        p.error('positive gene/batch counts and at least 2 bootstraps required')
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    cache = out/'prepared.pkl'
    spec = (str(Path(args.data).resolve()), args.cell, args.q)
    t0=time.perf_counter()
    if cache.exists():
        with cache.open('rb') as f: saved_spec,a,treatment,covariate = pickle.load(f)
        if saved_spec != spec: raise ValueError('Prepared cache differs: use another --out directory')
    else:
        a,treatment,covariate=prepare(args.data,args.cell,args.q)
        with cache.open('wb') as f: pickle.dump((spec,a,treatment,covariate),f)
    prep_seconds=time.perf_counter()-t0
    groups=a.uns['memento']['groups']
    average = np.mean([a.uns['memento']['1d_moments'][g][0] for g in groups],axis=0)
    order = np.argsort(average)
    indices = order[np.linspace(0,len(order)-1,min(args.genes,len(order))).astype(int)]
    genes = a.var_names[indices].tolist()
    assignment = {gene: ['stim'] for gene in genes}
    report = {'args':vars(args), 'cpu_affinity':sorted(os.sched_getaffinity(0)),
              'torch':torch.__version__, 'gpu':torch.cuda.get_device_name(0),
              'cells':a.n_obs,'eligible_genes':a.n_vars, 'tested_genes':len(genes),
              'group_cells':{g:a.uns['memento']['group_cells'][g].shape[0] for g in groups},
              'preparation_seconds':prep_seconds,'runs':{}}
    print(json.dumps({k:v for k,v in report.items() if k!='args'},indent=2),flush=True)
    # Warm CUDA before timing; the public call itself includes task construction,
    # state compression, copies, sampling, transformations, and regression.
    torch.ones(2,device='cuda').sum().item()
    results={}
    for mode in args.modes:
        if mode not in ('cpu','cpu_repeat','gpu_cpu','gpu_cpu_map','gpu'):
            raise ValueError(mode)
        torch.manual_seed(42)  # shared draws for hybrid and resident paths
        torch.cuda.reset_peak_memory_stats()
        kwargs=dict(treatment=treatment,covariate=covariate,treatment_for_gene=assignment,
                    num_boot=args.boots,num_cpus=args.cpu_jobs,verbose=0,
                    random_state=6 if mode=='cpu_repeat' else 5)
        start=time.perf_counter()
        if mode.startswith('cpu'):
            memento.ht_1d_moments(a,**kwargs)
            extra={}
        else:
            regression = {'gpu_cpu':'cpu', 'gpu_cpu_map':'cpu_map', 'gpu':'gpu'}[mode]
            dispatch=GPUDispatch(args.batch, regression,args.validate)
            with gpu_execution(dispatch): memento.ht_1d_moments(a,**kwargs)
            extra={'stages_seconds':dispatch.timings,
                   'padding_fraction':1-dispatch.padding_real/dispatch.padding_total,
                   'same_input_max_abs_errors':dispatch.checks}
        torch.cuda.synchronize()
        elapsed=time.perf_counter()-start
        result={k:np.asarray(v).copy() for k,v in a.uns['memento']['1d_ht'].items()
                if k in ('mean_coef','mean_se','mean_asl','var_coef','var_se','var_asl')}
        results[mode]=result
        np.savez(out/f'{mode}_{len(genes)}g_{args.boots}b.npz',genes=np.array(genes),**result)
        report['runs'][mode]={'wall_seconds':elapsed, 'peak_allocated_mb':torch.cuda.max_memory_allocated()/1024**2,
                              **extra}
        print(mode,json.dumps(report['runs'][mode]),flush=True)
        with (out/f'report_{len(genes)}g_{args.boots}b.json').open('w') as f: json.dump(report,f,indent=2)
    report['comparisons']={}
    for left,right in [('cpu','gpu'),('cpu','cpu_repeat'),('gpu_cpu','gpu')]:
        if left in results and right in results:
            report['comparisons'][f'{left}_vs_{right}']=comparison(results[left],results[right])
    with (out/f'report_{len(genes)}g_{args.boots}b.json').open('w') as f: json.dump(report,f,indent=2)
    print(json.dumps(report['comparisons'],indent=2),flush=True)


if __name__ == '__main__':
    main()
