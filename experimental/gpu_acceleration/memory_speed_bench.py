"""Compare the original GPU path, cross-group batching, and CUDA graph replay."""
import os
import argparse
import gc
import json
import pickle
import time
from pathlib import Path
from unittest.mock import patch
from real_data_bench import np, torch, memento, GPUDispatch, gpu_execution, comparison
from gpu_1d_memory import WideGPUDispatch, MixedSampler
from gpu_states import GPUStateDispatch
from memento import bootstrap, estimator


def validate_sampler():
    states=[]
    qs=[0.,.07,.2,.4]
    for counts in ([10,30,60],[2,8],[100],np.ones(70,dtype=int)):
        counts=np.array(counts)
        e=np.linspace(0,7,len(counts))[:,None]
        inv=np.linspace(.5,1.3,len(counts))[:,None]
        states.append((inv,inv**2,e,counts))
    eager=MixedSampler(False); graph=MixedSampler(True); exact=MixedSampler(True,True)
    b=40000
    em,ev,w=eager(states,qs,b,True)
    ns=np.array([s[3].sum() for s in states])
    np.testing.assert_array_equal(w.sum(0).cpu().numpy(),np.broadcast_to(ns[:,None],(4,b)))
    gm,gv=graph(states,qs,b)
    assert bool((graph.last_remaining==0).all())
    first=gm.clone()
    gm2,_=graph(states,qs,b)
    assert not torch.equal(first,gm2), 'Graph replay must advance RNG'
    assert torch.equal(gm,first), 'Returned samples must not alias graph workspace'
    xm,xv=exact(states,qs,b)
    assert bool((exact.last_remaining==0).all())
    xm2,_=exact(states,qs,b)
    assert not torch.equal(xm,xm2), 'Exact-shape graph replay must advance RNG'
    report=[]
    rng=np.random.default_rng(104)
    for i,(inv,inv2,e,counts) in enumerate(states):
        k=len(counts); n=counts.sum(); p=counts/n
        weights=w[:k,i].double().cpu().numpy()
        expected=n*(np.diag(p)-np.outer(p,p))
        if k<10:
            np.testing.assert_allclose(np.atleast_2d(np.cov(weights,bias=True)),expected,rtol=.05,atol=.04)
        cpu_w=rng.multinomial(n,p,size=b).T
        cm,cv=estimator._hyper_1d_relative((e,cpu_w),n,qs[i],(inv,inv2))
        exact_m,exact_v=estimator._hyper_1d_relative((e,weights),n,qs[i],(inv,inv2))
        np.testing.assert_allclose(em[i].cpu().numpy(),exact_m,rtol=2e-5,atol=2e-5)
        np.testing.assert_allclose(ev[i].cpu().numpy(),exact_v,rtol=2e-5,atol=2e-5)
        if k>1:
            ratios=[float(gm[i].std(correction=0).item()/cm.std()),float(gv[i].std(correction=0).item()/cv.std())]
            np.testing.assert_allclose(ratios,1,rtol=.04)
            exact_ratios=[float(xm[i].std(correction=0).item()/cm.std()),float(xv[i].std(correction=0).item()/cv.std())]
            np.testing.assert_allclose(exact_ratios,1,rtol=.04)
            report.append({'cells':int(n),'states':k,'q':qs[i],'graph_sd_ratios':ratios,
                           'exact_graph_sd_ratios':exact_ratios})
        else:
            np.testing.assert_allclose(gm[i].cpu().numpy(),cm)
            np.testing.assert_allclose(gv[i].cpu().numpy(),cv)
    del eager,graph,exact,em,ev,gm,gv,w,gm2,first,xm,xv,xm2
    gc.collect(); torch.cuda.empty_cache()
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--genes',type=int,default=1742)
    p.add_argument('--boots',type=int,default=10000)
    p.add_argument('--batch',type=int,default=256)
    p.add_argument('--row-batch',type=int,default=1024)
    p.add_argument('--sampler-mib',type=int,default=256)
    p.add_argument('--modes',nargs='+',default=['baseline','wide','graph','wide','baseline','graph'])
    p.add_argument('--validate',action='store_true')
    p.add_argument('--out',default='experimental/gpu_acceleration/results_memory_speed')
    args=p.parse_args()
    if min(args.genes,args.boots,args.batch,args.row_batch,args.sampler_mib)<1:
        p.error('All numeric parameters must be positive')
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    report={'args':vars(args),'cpu_affinity':sorted(os.sched_getaffinity(0)),
            'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,'runs':[]}
    report['sampler_checks']=validate_sampler()
    print('Sampler checks passed',flush=True)
    with open('experimental/gpu_acceleration/results/prepared.pkl','rb') as f:
        spec,a,treatment,covariate=pickle.load(f)
    u=a.uns['memento']
    order=np.argsort(np.mean([u['1d_moments'][g][0] for g in u['groups']],axis=0))
    take=order[np.linspace(0,len(order)-1,min(args.genes,len(order))).astype(int)]
    genes=a.var_names[take].tolist()
    assignment={g:['stim'] for g in genes}
    report.update(dataset=spec,genes=len(genes),boots=args.boots)
    kwargs=dict(treatment=treatment,covariate=covariate,treatment_for_gene=assignment,
                num_boot=args.boots,num_cpus=1,verbose=0,random_state=5)
    # Graph objects persist for their warm repeat; no states or samples are cached.
    graph_dispatches={}
    outputs={}
    for trial,mode in enumerate(args.modes):
        if mode=='baseline':
            dispatch=GPUDispatch(args.batch,validate=args.validate)
        elif mode=='wide':
            dispatch=WideGPUDispatch(args.batch,args.row_batch,False,args.validate)
        elif mode=='gpu_states':
            dispatch=GPUStateDispatch(args.batch,args.row_batch,args.validate)
        elif mode in ('graph','graph_exact'):
            if mode not in graph_dispatches:
                graph_dispatches[mode]=WideGPUDispatch(args.batch,args.row_batch,True,args.validate,
                                                      graph_exact=mode=='graph_exact')
            dispatch=graph_dispatches[mode]
        else:
            raise ValueError(mode)
        before=dispatch.timings.copy()
        check_start=len(dispatch.checks)
        capture_before=dispatch.sampler.capture_seconds if isinstance(dispatch,WideGPUDispatch) else 0
        gc.collect(); torch.cuda.empty_cache(); torch.manual_seed(42)
        torch.cuda.reset_peak_memory_stats()
        start=time.perf_counter()
        # Original baseline retains its original 64 MiB workspace. Wide/graph
        # share the enlarged budget, isolating graph replay from extra memory.
        budget=64 if mode=='baseline' else args.sampler_mib
        with patch.object(bootstrap,'_DEFAULT_BOOTSTRAP_BATCH_BYTES',budget*1024**2),gpu_execution(dispatch):
            memento.ht_1d_moments(a,**kwargs)
        torch.cuda.synchronize()
        elapsed=time.perf_counter()-start
        result={k:np.asarray(u['1d_ht'][k]).copy() for k in
                ('mean_coef','mean_se','mean_asl','var_coef','var_se','var_asl')}
        assert all(np.isfinite(v).all() for v in result.values())
        row={'mode':mode,'trial':trial,'wall_seconds':elapsed,
             'stages_seconds':{k:dispatch.timings[k]-before[k] for k in before},
             'peak_allocated_gib':torch.cuda.max_memory_allocated()/1024**3,
             'peak_reserved_gib':torch.cuda.max_memory_reserved()/1024**3,
             'regression_checks':dispatch.checks[check_start:]}
        if isinstance(dispatch,WideGPUDispatch):
            row['capture_seconds']=dispatch.sampler.capture_seconds-capture_before
            row['cached_graphs']=len(dispatch.sampler.cache)
            row['graph_replays_cumulative']=dispatch.sampler.replays
        if 'baseline' in outputs:
            row['vs_baseline']=comparison(outputs['baseline'],result)
        if mode=='gpu_states' and 'wide' in outputs:
            for key,value in result.items():
                np.testing.assert_array_equal(value,outputs['wide'][key])
            row['same_seed_wide_equal']=True
        outputs[mode]=result
        np.savez(out/f'{trial}_{mode}.npz',genes=np.array(genes),**result)
        report['runs'].append(row)
        (out/'report.json').write_text(json.dumps(report,indent=2))
        print(json.dumps({k:v for k,v in row.items() if k not in ('vs_baseline','regression_checks')}),flush=True)
    print(out/'report.json',flush=True)


if __name__=='__main__': main()
