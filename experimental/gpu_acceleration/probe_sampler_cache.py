"""Inventory cache sizes and observed binomial inputs; no custom sampler.

Timings include instrumentation and must not be used as performance estimates.
"""
import os
import argparse
import json
import pickle
import time
from pathlib import Path
from real_data_bench import np, torch
from gpu_states import compress_group


def summarize(values):
    return dict(zip(('min','p50','p90','p99','max'),
                    np.quantile(values,[0,.5,.9,.99,1]).tolist())) if len(values) else {}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--genes',type=int,default=64)
    parser.add_argument('--boots',type=int,default=10000)
    parser.add_argument('--seed',type=int,default=421)
    parser.add_argument('--out',default='experimental/gpu_acceleration/results_sampler_cache')
    args=parser.parse_args()
    if min(args.genes,args.boots)<1: parser.error('genes and boots must be positive')
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    with open('experimental/gpu_acceleration/results/prepared.pkl','rb') as f:
        spec,a,_,_=pickle.load(f)
    u=a.uns['memento']
    order=np.argsort(np.mean([u['1d_moments'][g][0] for g in u['groups']],axis=0))
    selected=order[np.linspace(0,len(order)-1,min(args.genes,len(order))).astype(int)]
    torch.manual_seed(args.seed)
    # Global exact-fp32 probability keys. Complement is rounded to fp32 as in
    # sample_binomial<float>; this is an inventory, not a precision change.
    probability_max_n={}
    pair_frequencies={}
    branches={'deterministic':0,'small_mean':0,'btrs':0}
    occupied=[];cover90=[];cover99=[];repeat_counts=[]
    inventory_states=0;inventory_rows=0;naive_entries=0
    sampled_states=0;sampled_rows=0;sampled_nontrivial_draws=0
    sampled_dense_entries=0;groups=[]
    started=time.perf_counter()
    for group in u['groups']:
        dense=u['group_cells'][group].toarray().T
        n=dense.shape[1]
        packed=compress_group(dense,u['approx_size_factor'][group],
                              np.full(len(dense),u['group_q'][group]))
        probabilities=packed.coeff[0].cpu().numpy()
        mu,_,rv=u['1d_moments'][group]
        valid=~(np.isnan(mu)|np.isnan(rv)|(mu==0)|(rv<0)) & (packed.sizes>1)
        for gene in np.flatnonzero(valid):
            lo=int(packed.starts[gene]);hi=lo+int(packed.sizes[gene])
            p=probabilities[lo:hi]
            q=np.minimum(p,np.float32(1)-p)
            q=q[(q>0)&(q<1)]
            inventory_states+=len(q);inventory_rows+=1
            naive_entries+=len(q)*(n+1)
            for bits in np.unique(q.view(np.uint32)):
                key=int(bits)
                probability_max_n[key]=max(n,probability_max_n.get(key,0))
        take=selected[valid[selected]]
        if not len(take): continue
        sizes=packed.sizes[take]
        k=int(sizes.max());g=len(take)
        p=np.zeros((k,g),np.float32)
        for col,gene in enumerate(take):
            lo=int(packed.starts[gene]);size=int(packed.sizes[gene])
            p[:size,col]=probabilities[lo:lo+size]
        p_gpu=torch.as_tensor(p,device='cuda')
        rem=torch.full((g,args.boots),float(n),device='cuda')
        hist=torch.empty((k,g,n+1),dtype=torch.int32,device='cuda')
        offsets=torch.arange(g,device='cuda')[:,None]*(n+1)
        for j in range(k):
            hist[j]=torch.bincount((rem.long()+offsets).flatten(),
                                  minlength=g*(n+1)).reshape(g,n+1).int()
            draw=torch.binomial(rem,p_gpu[j,:,None].expand_as(rem))
            rem.sub_(draw)
        assert bool((rem==0).all()), 'Count conservation failed'
        hist=hist.cpu().numpy()
        remaining=np.arange(n+1)
        local_branches={key:0 for key in branches}
        for col,gene in enumerate(take):
            for j in range(int(sizes[col])):
                h=hist[j,col]
                assert int(h.sum())==args.boots
                prob=p[j,col];q=np.minimum(prob,np.float32(1)-prob)
                if q<=0:
                    local_branches['deterministic']+=int(h.sum())
                    continue
                sampled_states+=1
                sampled_dense_entries+=n+1
                local_branches['deterministic']+=int(h[0])
                # Scalar float multiplication in the installed PyTorch branch.
                use_btrs=remaining.astype(np.float32)*q>=np.float32(10)
                local_branches['btrs']+=int(h[use_btrs].sum())
                local_branches['small_mean']+=int(h[(remaining>0)&~use_btrs].sum())
                nonzero=np.flatnonzero(h[1:])+1
                freq=h[nonzero]
                if not len(freq): continue
                occupied.append(len(freq));repeat_counts.extend(freq.tolist())
                ranked=np.sort(freq)[::-1].cumsum()
                cover90.append(int(np.searchsorted(ranked,.90*ranked[-1])+1))
                cover99.append(int(np.searchsorted(ranked,.99*ranked[-1])+1))
                bits=int(np.asarray(q).view(np.uint32))
                for r,freq_r in zip(nonzero,freq):
                    key=(bits,int(r))
                    pair_frequencies[key]=pair_frequencies.get(key,0)+int(freq_r)
        for key,value in local_branches.items(): branches[key]+=value
        sampled_rows+=len(take)
        sampled_nontrivial_draws+=local_branches['small_mean']+local_branches['btrs']
        groups.append({'group':str(group),'cells':n,'sampled_rows':len(take),
                       'branch_draw_counts':local_branches})
        print(json.dumps(groups[-1]),flush=True)
        del packed,p_gpu,rem,hist
    assert sum(pair_frequencies.values())==sampled_nontrivial_draws
    dedup_entries=sum(n+1 for n in probability_max_n.values())
    btrs_entries=0
    for bits,n in probability_max_n.items():
        q=np.asarray(bits,dtype=np.uint32).view(np.float32)
        btrs_entries+=int(np.count_nonzero(np.arange(n+1,dtype=np.float32)*q>=10))
    keys=sorted(pair_frequencies)
    pbits=np.array([key[0] for key in keys],dtype=np.uint32)
    counts=np.array([key[1] for key in keys],dtype=np.int32)
    frequencies=np.array([pair_frequencies[key] for key in keys],dtype=np.int64)
    # Empirical input distribution for a later microbenchmark. This does not
    # preserve launch order or within-warp input correlations.
    np.savez(out/'observed_inputs.npz',probability=pbits.view(np.float32),
             remaining=counts,frequency=frequencies,selected_gene_indices=selected)
    report={
        'args':vars(args),'dataset':spec,'torch':torch.__version__,
        'gpu':torch.cuda.get_device_name(0),'cpu_affinity':sorted(os.sched_getaffinity(0)),
        'scope':'Full eligible-state inventory; expression-stratified subset of genes, all groups, for observed inputs. Padding excluded. Counts are draw-weighted, not time-weighted.',
        'inventory':{'eligible_gene_group_rows':inventory_rows,'nontrivial_state_rows':inventory_states,
                     'distinct_folded_fp32_probabilities':len(probability_max_n),
                     'probability_cache_two_fp64_fields_bytes':len(probability_max_n)*16,
                     'per_state_probability_cache_two_fp64_fields_bytes':inventory_states*16,
                     'naive_per_state_remaining_entries':naive_entries,
                     'naive_eight_fp64_fields_bytes':naive_entries*64,
                     'deduplicated_remaining_entries':dedup_entries,
                     'deduplicated_eight_fp64_fields_bytes':dedup_entries*64,
                     'deduplicated_btrs_entries':btrs_entries,
                     'deduplicated_btrs_eight_fp64_fields_bytes':btrs_entries*64},
        'observed':{'sampled_gene_group_rows':sampled_rows,'nontrivial_state_rows':sampled_states,
                    'branch_draw_counts':branches,
                    'branch_draw_fractions':{key:value/sum(branches.values()) for key,value in branches.items()},
                    'positive_remaining_values_per_state':summarize(occupied),
                    'entries_for_90_percent_draws_per_state':summarize(cover90),
                    'entries_for_99_percent_draws_per_state':summarize(cover99),
                    'draws_per_observed_state_remaining_entry':summarize(repeat_counts),
                    'per_state_observed_entries':sum(occupied),
                    'per_state_dense_entries':sampled_dense_entries,
                    'distinct_global_probability_remaining_pairs':len(keys),
                    'global_draws_per_pair':summarize(frequencies),
                    'nontrivial_draws':sampled_nontrivial_draws},
        'groups':groups,
        'instrumented_wall_seconds':time.perf_counter()-started,
        'limitations':['No cache lookup, custom kernel, or speed comparison implemented.',
                       'Observed entries cannot safely define complete future support; cache misses need fallback.',
                       'Table byte estimates exclude indices, alignment and construction temporaries; eight fp64 fields is a budgeting assumption, not an implemented layout.',
                       'Subsample trajectories use the existing sampler but different batching/RNG order from the full dispatcher.'],
    }
    (out/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({key:report[key] for key in ('inventory','observed','instrumented_wall_seconds')},indent=2),flush=True)


if __name__=='__main__': main()
