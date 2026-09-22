"""CUDA-event timings for a representative real-state sampler.

Intervals include any host launch gaps between events. This is not a CUPTI
kernel-only trace; CUPTI profiling stalled in this WSL environment.
"""
import os
import json
import pickle
from pathlib import Path
from real_data_bench import np, torch
from gpu_1d_memory import MixedSampler, pack_states
from memento import bootstrap


def main():
    with open('experimental/gpu_acceleration/results/prepared.pkl','rb') as f:
        _,a,_,_=pickle.load(f)
    u=a.uns['memento']
    group=max(u['groups'],key=lambda g:u['group_cells'][g].shape[0])
    states=[]
    for i in range(a.n_vars):
        state=bootstrap._unique_expr(u['group_cells'][group][:,i],u['approx_size_factor'][group])
        if 64<=len(state[3])<=127:
            states.append(state)
        if len(states)==256:
            break
    sampler=MixedSampler()
    qs=[u['group_q'][group]]*len(states)
    for _ in range(2): sampler(states,qs,10000)
    torch.cuda.synchronize()
    packed,ns=pack_states(states,qs)
    coeff=torch.as_tensor(packed,device='cuda')
    ns_t=torch.as_tensor(ns,device='cuda')[:,None]
    k=packed.shape[1]
    events=[[torch.cuda.Event(enable_timing=True) for _ in range(3)] for _ in range(k)]
    rows=[]
    for repeat in range(3):
        rem=ns_t.expand(len(states),10000).clone()
        m1=torch.zeros_like(rem); m2=torch.zeros_like(rem)
        for j,(start,middle,end) in enumerate(events):
            start.record()
            d=torch.binomial(rem,coeff[0,j,:,None].expand_as(rem))
            middle.record()
            rem.sub_(d)
            m1.addcmul_(coeff[1,j,:,None],d)
            m2.addcmul_(coeff[2,j,:,None],d)
            end.record()
        torch.cuda.synchronize()
        binomial=sum(start.elapsed_time(middle) for start,middle,end in events)
        update=sum(middle.elapsed_time(end) for start,middle,end in events)
        rows.append({'repeat':repeat,'binomial_ms':binomial,'moment_update_ms':update,
                     'binomial_fraction':binomial/(binomial+update)})
    out=Path('experimental/gpu_acceleration/results_memory_speed/profile.json')
    out.parent.mkdir(parents=True,exist_ok=True)
    report={'group':group,'cells':u['group_cells'][group].shape[0],
            'gene_group_rows':len(states),'boots':10000,'max_states':max(len(s[3]) for s in states),
            'method':'CUDA events; includes host launch gaps; excludes preparation/transfers',
            'timings':rows}
    out.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
