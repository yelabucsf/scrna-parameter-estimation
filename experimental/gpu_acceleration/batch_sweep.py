"""Repeated full-workload batch sweep with the same two-CPU resource limit."""
import os
import argparse
import gc
import json
import pickle
import time
from pathlib import Path
from unittest.mock import patch

# Imports the repository and applies the same torch/thread configuration.
from real_data_bench import np, torch, memento, GPUDispatch, gpu_execution
from memento import bootstrap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepared', default='experimental/gpu_acceleration/results/prepared.pkl')
    parser.add_argument('--batches', nargs='+', type=int, default=[256, 512, 1024, 1742])
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--boots', type=int, default=10000)
    parser.add_argument('--sampler-mib', type=int, default=64,
                        help='Scoped experiment override of the existing bootstrap working-array budget')
    parser.add_argument('--out', default='experimental/gpu_acceleration/results_batch_sweep/report.json')
    args = parser.parse_args()
    if min(args.batches) < 1 or args.repeats < 1 or args.boots < 2 or args.sampler_mib < 1:
        parser.error('positive batches/repeats and at least two bootstraps required')
    with open(args.prepared, 'rb') as f:
        spec, a, treatment, covariate = pickle.load(f)
    u = a.uns['memento']
    average = np.mean([u['1d_moments'][g][0] for g in u['groups']], axis=0)
    genes = a.var_names[np.argsort(average)].tolist()
    assignment = {g: ['stim'] for g in genes}
    kwargs = dict(treatment=treatment, covariate=covariate, num_boot=args.boots,
                  num_cpus=1, verbose=0, random_state=5)
    # Warm all execution stages on a small real workload, outside measurements.
    with patch.object(bootstrap, '_DEFAULT_BOOTSTRAP_BATCH_BYTES', args.sampler_mib*1024**2), gpu_execution(GPUDispatch(32)):
        memento.ht_1d_moments(a, treatment_for_gene={g:['stim'] for g in genes[::max(1,len(genes)//32)][:32]},
                              **{**kwargs, 'num_boot':1000})
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    report = {'args':vars(args), 'dataset':spec, 'genes':len(genes), 'cells':a.n_obs,
              'groups':len(u['groups']), 'torch':torch.__version__,
              'gpu':torch.cuda.get_device_name(0),
              'cpu_affinity':sorted(os.sched_getaffinity(0)),
              'total_vram_gib':total/1024**3, 'initial_free_vram_gib':free/1024**3,
              'runs':[]}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for repeat in range(args.repeats):
        # Reverse the order on alternate passes to reveal time/order effects.
        batches = args.batches if repeat % 2 == 0 else args.batches[::-1]
        for batch in batches:
            gc.collect()
            torch.cuda.empty_cache()
            torch.manual_seed(42)
            torch.cuda.reset_peak_memory_stats()
            row = {'repeat':repeat, 'batch':batch}
            dispatch = GPUDispatch(batch)
            start = time.perf_counter()
            try:
                with patch.object(bootstrap, '_DEFAULT_BOOTSTRAP_BATCH_BYTES', args.sampler_mib*1024**2), gpu_execution(dispatch):
                    memento.ht_1d_moments(a, treatment_for_gene=assignment, **kwargs)
                torch.cuda.synchronize()
                row.update(wall_seconds=time.perf_counter()-start,
                    peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
                    peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3,
                    stages_seconds=dispatch.timings,
                    padding_fraction=1-dispatch.padding_real/dispatch.padding_total)
                summaries = {k:np.asarray(u['1d_ht'][k]).copy() for k in
                    ('mean_coef','mean_se','mean_asl','var_coef','var_se','var_asl')}
                row['nonfinite'] = {k:int((~np.isfinite(v)).sum()) for k,v in summaries.items()}
                assert all(n == 0 for n in row['nonfinite'].values())
                np.savez(out.parent/f'batch{batch}_repeat{repeat}.npz',genes=np.array(genes),**summaries)
            except torch.cuda.OutOfMemoryError:
                row['error'] = 'CUDA out of memory'
                row['wall_seconds'] = time.perf_counter()-start
            report['runs'].append(row)
            out.write_text(json.dumps(report, indent=2))
            print(json.dumps(row), flush=True)
    report['summary'] = {}
    for batch in args.batches:
        rows = [r for r in report['runs'] if r['batch']==batch and 'error' not in r]
        if rows:
            times = [r['wall_seconds'] for r in rows]
            report['summary'][str(batch)] = {
                'median_seconds':float(np.median(times)), 'range_seconds':[min(times),max(times)],
                'peak_allocated_gib':max(r['peak_allocated_gib'] for r in rows),
                'peak_reserved_gib':max(r['peak_reserved_gib'] for r in rows)}
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report['summary'], indent=2), flush=True)


if __name__ == '__main__':
    main()
