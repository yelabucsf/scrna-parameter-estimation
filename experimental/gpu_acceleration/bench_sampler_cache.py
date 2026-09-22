"""Standalone probability-cache experiment; does not modify ht_1d_moments.

Predeclared checks: exact cached/header-reference equality at equal RNG inputs;
mean and variance errors <6 Monte Carlo standard errors; grouped PMF chi-square
p>=1e-6. These also apply to the independent torch.binomial control.
"""
import os
import argparse
import itertools
import json
import pickle
import time
from pathlib import Path
from real_data_bench import np, torch
from scipy.stats import binom, chi2
from gpu_states import compress_group
from sampler_cache_cuda import CudaSampler

MODES=('direct','state_cache','shared_cache')


def sync(): torch.cuda.synchronize()


def phi(seed,stream,block):
    mask=(1<<32)-1
    c=[block&mask,block>>32,stream&mask,stream>>32]
    key=[seed&mask,seed>>32]
    for _ in range(10):
        a=0xD2511F53*c[0];b=0xCD9E8D57*c[2]
        c=[((b>>32)^c[1]^key[0])&mask,b&mask,
           ((a>>32)^c[3]^key[1])&mask,a&mask]
        key=[(key[0]+0x9E3779B9)&mask,(key[1]+0xBB67AE85)&mask]
    return c


def statistical_check(samples,cases):
    rows=[]
    for x,(n,p) in zip(samples,cases):
        p=float(np.float32(p));b=len(x)
        assert np.all((x>=0)&(x<=n)&(x==np.floor(x)))
        mean=n*p;var=n*p*(1-p)
        if not var:
            assert np.all(x==mean)
            rows.append({'n':n,'p':p,'deterministic':True});continue
        zmean=float((x.mean(dtype=np.float64)-mean)/np.sqrt(var/b))
        empirical_var=float(np.var(x,dtype=np.float64,ddof=1))
        fourth=3*var**2+var*(1-6*p*(1-p))
        zvar=float((empirical_var-var)/np.sqrt((fourth-(b-3)/(b-1)*var**2)/b))
        observed=np.bincount(x.astype(np.int64),minlength=n+1)
        expected=binom.pmf(np.arange(n+1),n,p)*b
        # Pool adjacent support points until expected bin count >=20; append
        # the last sparse tail to the preceding bin. No post-hoc tuning.
        pairs=[];obs=0;exp=0.
        for o,e in zip(observed,expected):
            obs+=int(o);exp+=float(e)
            if exp>=20: pairs.append([obs,exp]);obs=0;exp=0.
        if pairs:
            pairs[-1][0]+=obs;pairs[-1][1]+=exp
        else: pairs=[[obs,exp]]
        statistic=sum((o-e)**2/e for o,e in pairs if e>0)
        pvalue=float(chi2.sf(statistic,len(pairs)-1)) if len(pairs)>1 else None
        assert abs(zmean)<6,(n,p,'mean',zmean)
        assert abs(zvar)<6,(n,p,'variance',zvar)
        assert pvalue is None or pvalue>=1e-6,(n,p,'pmf',pvalue)
        rows.append({'n':n,'p':p,'mean_z':zmean,'variance_z':zvar,'pmf_p':pvalue,
                     'pmf_bins':len(pairs)})
    return rows


def validate(k):
    for seed in (0,1234567890123):
        words=torch.empty((8,12),device='cuda',dtype=torch.int32)
        k.rng(words,seed)
        expected=np.array([[word for block in range(3) for word in phi(seed,i,block)]
                           for i in range(8)],dtype=np.uint32)
        got=words.cpu().numpy().view(np.uint32)
        np.testing.assert_array_equal(got,expected)
        if seed==0:
            np.testing.assert_array_equal(got[0,:4],np.array(
                [0x6627e8d5,0xe169c58d,0xbc57ac4c,0x9b00dbd8],dtype=np.uint32))
    cases=[(0,.2),(1,0),(1,1),(1,.2),(7,.1),(7,.5),(7,.9)]
    cases += [(n,p) for n in (112,778) for p in (0,1e-6,.001,.01,.1,.5,.9,.999999,1)]
    cases += [(100,p) for p in (.099999,.1,.100001,.899999,.9,.900001)]
    ns=torch.tensor([n for n,p in cases],device='cuda',dtype=torch.float32)
    p=torch.tensor([p for n,p in cases],device='cuda',dtype=torch.float32)
    counts=ns[:,None].expand(-1,200000).contiguous()
    logs=torch.empty_like(p);k.logs(p,logs)
    q=torch.minimum(p,1-p)
    unique,ids=torch.unique(q,sorted=True,return_inverse=True);ids=ids.int()
    shared=torch.empty_like(unique);k.logs(unique,shared)
    reference=torch.empty_like(counts);output=torch.empty_like(counts)
    k.sample('header_reference',counts,p,logs,ids,reference,912)
    equal={}
    for mode in MODES:
        k.sample(mode,counts,p,shared if mode=='shared_cache' else logs,ids,output,912)
        assert torch.equal(reference,output),mode
        equal[mode]=True
    direct_checks=statistical_check(reference.cpu().numpy(),cases)
    torch.manual_seed(913)
    control=torch.binomial(counts,p[:,None].expand_as(counts))
    torch_checks=statistical_check(control.cpu().numpy(),cases)
    k.sample('direct',counts,p,logs,ids,output,914)
    assert not torch.equal(reference,output),'Fresh seeds must change nondegenerate draws'
    return {'cases':len(cases),'draws_per_case':200000,'exact_vs_header':equal,
            'philox_integer_reference_passed':True,'fresh_seed_passed':True,
            'direct_distribution_checks':direct_checks,'torch_control_checks':torch_checks,
            'thresholds':{'absolute_monte_carlo_z':6,'min_grouped_pmf_p':1e-6}}


def capture(rows,boots,genes,seed):
    with open('experimental/gpu_acceleration/results/prepared.pkl','rb') as f:
        spec,a,_,_=pickle.load(f)
    u=a.uns['memento'];rng=np.random.default_rng(seed)
    order=np.argsort(np.mean([u['1d_moments'][g][0] for g in u['groups']],axis=0))
    selected=order[np.linspace(0,len(order)-1,min(genes,len(order))).astype(int)]
    plans=[];inventory=[];slots=[]
    for gi,group in enumerate(u['groups']):
        dense=u['group_cells'][group].toarray().T;n=dense.shape[1]
        packed=compress_group(dense,u['approx_size_factor'][group],np.full(len(dense),u['group_q'][group]))
        p=packed.coeff[0].cpu().numpy();mu,_,rv=u['1d_moments'][group]
        valid=~(np.isnan(mu)|np.isnan(rv)|(mu==0)|(rv<0))&(packed.sizes>1)
        for gene in np.flatnonzero(valid):
            start=int(packed.starts[gene]);size=int(packed.sizes[gene])
            values=p[start:start+size-1]  # last state is deterministic p=1
            inventory.append(values.copy())
        take=selected[valid[selected]];sizes=packed.sizes[take]
        probs=np.zeros((int(sizes.max()),len(take)),np.float32)
        for col,gene in enumerate(take):
            start=int(packed.starts[gene]);size=int(sizes[col])
            probs[:size,col]=p[start:start+size]
            slots.extend((gi,j,col) for j in range(size-1))
        plans.append((str(group),n,probs))
    chosen=np.sort(rng.choice(len(slots),size=min(rows,len(slots)),replace=False))
    chosen=[slots[i] for i in chosen]
    result_counts=[];result_p=[];metadata=[]
    torch.manual_seed(seed)
    for gi,(group,n,p) in enumerate(plans):
        selected_slots=[(j,col) for group_id,j,col in chosen if group_id==gi]
        if not selected_slots: continue
        remaining=torch.full((p.shape[1],boots),float(n),device='cuda')
        pgpu=torch.as_tensor(p,device='cuda')
        recorded=torch.empty((len(selected_slots),boots),device='cuda')
        for j in range(len(p)):
            pick=[(i,col) for i,(state,col) in enumerate(selected_slots) if state==j]
            if pick:
                dest,source=zip(*pick)
                recorded[list(dest)]=remaining[list(source)]
            remaining.sub_(torch.binomial(remaining,pgpu[j,:,None].expand_as(remaining)))
        assert bool((remaining==0).all())
        result_counts.append(recorded.cpu().numpy())
        result_p.extend(p[j,col] for j,col in selected_slots)
        metadata.append({'group':group,'cells':n,'captured_states':len(selected_slots)})
    return (np.concatenate(result_counts),np.array(result_p,dtype=np.float32),
            np.concatenate(inventory),{'dataset':spec,'groups':metadata,'genes':len(selected),
                                      'eligible_state_pool':len(slots),'captured_states':len(chosen)})


def setup(k,p,inventory):
    unique=torch.unique(torch.minimum(inventory,1-inventory),sorted=True)
    ids=torch.searchsorted(unique,torch.minimum(p,1-p)).int()
    logs=torch.empty_like(p);shared=torch.empty_like(unique)
    k.logs(p,logs);k.logs(unique,shared)
    return logs,shared,ids


def time_setup(k,p,inventory):
    rows=[]
    for _ in range(7):
        sync();start=time.perf_counter()
        all_logs=torch.empty_like(inventory);k.logs(inventory,all_logs)
        sync();state_ms=(time.perf_counter()-start)*1000
        start=time.perf_counter()
        unique,inverse=torch.unique(torch.minimum(inventory,1-inventory),sorted=True,return_inverse=True)
        full_ids=inverse.int()
        ids=torch.searchsorted(unique,torch.minimum(p,1-p)).int()
        shared=torch.empty_like(unique);k.logs(unique,shared)
        sync();shared_ms=(time.perf_counter()-start)*1000
        rows.append({'state_full_inventory_ms':state_ms,'shared_full_inventory_ms':shared_ms})
    return {'trials':rows,'inventory_states':inventory.numel(),
            'state_log_bytes':inventory.numel()*4,'shared_log_bytes':shared.numel()*4,
            'shared_full_state_mapping_bytes':full_ids.numel()*4,
            'shared_probabilities':shared.numel(),
            'note':'Wall time includes allocations, log kernels, shared-key sorting, full-inventory and trace ID mapping. Input probabilities already resident. Setup covers the full inventory, while kernel timings cover the sampled trace.'}


def benchmark(k,counts,p,inventory,repeats,seed):
    logs,shared,ids=setup(k,p,inventory)
    out=torch.empty_like(counts);reference=torch.empty_like(counts)
    for mode in MODES:
        for _ in range(3): k.sample(mode,counts,p,shared if mode=='shared_cache' else logs,ids,out,seed)
    k.sample('header_reference',counts,p,logs,ids,reference,seed)
    equal={}
    for mode in MODES:
        k.sample(mode,counts,p,shared if mode=='shared_cache' else logs,ids,out,seed)
        assert torch.equal(out,reference),(mode,'real trace equality')
        equal[mode]=True
    orders=list(itertools.permutations(MODES));rows=[]
    for trial in range(repeats):
        row={'trial':trial,'order':orders[trial%len(orders)]}
        for mode in row['order']:
            begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
            begin.record()
            k.sample(mode,counts,p,shared if mode=='shared_cache' else logs,ids,out,seed+trial+1)
            end.record();end.synchronize()
            row[mode+'_ms']=begin.elapsed_time(end)
        rows.append(row)
    median={mode:float(np.median([r[mode+'_ms'] for r in rows])) for mode in MODES}
    q=torch.minimum(p,1-p)[:,None]
    stochastic=(counts>0)&(q>0)
    btrs=stochastic&(counts*q>=10)
    branch_counts={'btrs':int(btrs.sum()),'small_mean':int((stochastic&~btrs).sum()),
                   'deterministic':int((~stochastic).sum())}
    return {'draws_per_launch':counts.numel(),'states':len(p),'boots_per_state':counts.shape[1],
            'branch_draw_counts':branch_counts,
            'exact_vs_header':equal,'trials':rows,'median_ms':median,
            'median_paired_speedup':{mode:float(np.median([r['direct_ms']/r[mode+'_ms'] for r in rows]))
                                   for mode in MODES if mode!='direct'},
            'paired_speedup_range':{mode:[float(min(r['direct_ms']/r[mode+'_ms'] for r in rows)),
                                         float(max(r['direct_ms']/r[mode+'_ms'] for r in rows))]
                                    for mode in MODES if mode!='direct'}}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--rows',type=int,default=1024)
    parser.add_argument('--boots',type=int,default=10000)
    parser.add_argument('--genes',type=int,default=64)
    parser.add_argument('--repeats',type=int,default=12)
    parser.add_argument('--seed',type=int,default=715)
    parser.add_argument('--out',default='experimental/gpu_acceleration/results_sampler_cache_bench')
    args=parser.parse_args()
    if min(args.rows,args.boots,args.genes,args.repeats)<1:parser.error('numeric parameters must be positive')
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter();k=CudaSampler();sync()
    report={'args':vars(args),'torch':torch.__version__,'gpu':torch.cuda.get_device_name(0),
            'cpu_affinity':sorted(os.sched_getaffinity(0)),
            'compile_seconds':time.perf_counter()-start,'source_hashes':k.source_hashes,
            'compile_options':k.options,'registers':k.registers,'compile_log':k.compile_log}
    print('Compiled',k.registers,flush=True)
    report['validation']=validate(k)
    print('Synthetic distribution and exact-equality checks passed',flush=True)
    counts_np,p_np,inventory_np,meta=capture(args.rows,args.boots,args.genes,args.seed)
    report['trace']=meta
    np.savez(out/'trace.npz',counts=counts_np,probability=p_np,inventory=inventory_np)
    counts=torch.as_tensor(counts_np,device='cuda');p=torch.as_tensor(p_np,device='cuda')
    inventory=torch.as_tensor(inventory_np,device='cuda')
    report['cache_setup']=time_setup(k,p,inventory)
    report['grouped']=benchmark(k,counts,p,inventory,args.repeats,args.seed)
    print('Grouped',report['grouped']['median_ms'],report['grouped']['median_paired_speedup'],flush=True)
    q=torch.minimum(p,1-p)
    only_small=(counts.max(dim=1).values*q<10)&(q>0)
    report['small_mean_only']=benchmark(k,counts[only_small].contiguous(),p[only_small].contiguous(),
                                        inventory,args.repeats,args.seed)
    print('Small-mean only',report['small_mean_only']['median_ms'],
          report['small_mean_only']['median_paired_speedup'],flush=True)
    # Stress case: mix individual draws from the actual trace. It intentionally
    # removes within-warp state coherence; not a model of the current dispatcher.
    torch.manual_seed(args.seed+1)
    ix=torch.randint(counts.numel(),(min(counts.numel(),2**20),),device='cuda')
    mixed_counts=counts.flatten()[ix][:,None].contiguous()
    mixed_p=p[ix//counts.shape[1]].contiguous()
    report['shuffled_stress']=benchmark(k,mixed_counts,mixed_p,inventory,args.repeats,args.seed)
    print('Shuffled',report['shuffled_stress']['median_ms'],report['shuffled_stress']['median_paired_speedup'],flush=True)
    report['limitations']=[
        'Standalone single-binomial launches; not a multinomial or end-to-end speed measurement.',
        'Same installed PyTorch Philox algorithm, but standalone stream assignment differs from torch.binomial.',
        'The grouped trace preserves shared probability across 10000 replicates; one thread handles one draw.',
        'Shared cache is warm; construction reported separately. No cold-device or eviction stress test.',
        'Kernel uses the installed binomial algorithm; only small-mean log1p is cached, BTRS is unchanged.',
        'Distribution tests are finite evidence, not proof or approval for production integration.']
    (out/'report.json').write_text(json.dumps(report,indent=2))
    print(out/'report.json',flush=True)


if __name__=='__main__':main()
