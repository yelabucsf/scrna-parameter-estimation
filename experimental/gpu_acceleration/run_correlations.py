"""Public GPU differential-correlation validation on a reproducible pair panel."""
import os
import argparse
import gc
import json
import sys
import time
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import anndata
import numpy as np
import pandas as pd
from scipy import sparse
from statsmodels.stats.multitest import multipletests
import torch
import memento
from memento import _gpu,hypothesis_test


def compare(a,b):
    j=a.merge(b,on=['gene_1','gene_2','tx'],suffixes=('_a','_b'))
    valid=np.isfinite(j.corr_se_a)&np.isfinite(j.corr_se_b)&(j.corr_se_a>0)
    ratio=j.loc[valid,'corr_se_b']/j.loc[valid,'corr_se_a']
    return dict(pairs=len(j),finite=int(valid.sum()),
        se_ratio_quantiles=np.quantile(ratio,[0,.05,.5,.95,1]).tolist(),
        max_coefficient_difference=float(abs(j.corr_coef_a-j.corr_coef_b).max()),
        nan_mismatch=int((j.corr_se_a.isna()!=j.corr_se_b.isna()).sum()))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',default='/mnt/c/Data/memento_workspace/interferon_filtered.h5ad')
    p.add_argument('--pairs',type=int,default=2000)
    p.add_argument('--boots',type=int,default=10000)
    p.add_argument('--cpu-pairs',type=int,default=64)
    p.add_argument('--out',default='experimental/gpu_acceleration/results_correlations')
    args=p.parse_args()
    torch.set_num_threads(2)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    source=anndata.read_h5ad(args.data,backed='r')
    start=time.perf_counter()
    a=source[source.obs.cell=='CD14+ Monocytes'].to_memory();source.file.close()
    a.X=sparse.csr_matrix(a.X);a.obs['capture_rate']=.07
    memento.setup_memento(a,q_column='capture_rate',min_cell_count=10)
    memento.create_groups(a,label_columns=['ind','cell','stim'])
    memento.compute_1d_moments(a,min_perc_group=.7)
    u=a.uns['memento'];groups=u['groups']
    meta=a.obs.drop_duplicates('memento_group').set_index('memento_group').loc[groups]
    tx=pd.DataFrame({'stim':(meta.stim=='stim').astype(float)},index=groups)
    cov=pd.get_dummies(meta.ind,drop_first=True,dtype=float)
    total=a.n_vars*(a.n_vars-1)//2
    if not 1<=args.pairs<=total:p.error(f'pairs must lie in [1,{total}]')
    # Uniform selection from all unordered nonself pairs, without replacement.
    rng=np.random.default_rng(29)
    ranks=np.sort(rng.choice(total,args.pairs,replace=False))
    ends=np.cumsum(np.arange(a.n_vars-1,0,-1))
    left=np.searchsorted(ends,ranks,side='right')
    starts=np.concatenate(([0],ends[:-1]))
    right=left+1+ranks-starts[left]
    pairs=list(zip(a.var_names[left],a.var_names[right]))
    pd.DataFrame(pairs,columns=['gene_1','gene_2']).to_csv(out/'pairs.csv',index=False)
    memento.compute_2d_moments(a,pairs)
    report=dict(data=args.data,cell_type='CD14+ Monocytes',cells=a.n_obs,genes=a.n_vars,
                pairs=len(pairs),possible_pairs=total,groups=len(groups),num_boot=args.boots,
                pair_selection='Uniform unordered nonself pairs without replacement, seed 29',
                design='Donor x cell-type x condition; stim vs ctrl, donor fixed effects',
                preparation_seconds=time.perf_counter()-start,q=.07,min_perc_group=.7,
                cpu_affinity=sorted(os.sched_getaffinity(0)),torch=torch.__version__,
                gpu=torch.cuda.get_device_name(0),gpu_runs=[])
    kwargs=dict(treatment=tx,covariate=cov,num_boot=args.boots,verbose=0,approx='norm',resample_rep=False)
    torch.ones(1,device='cuda').sum().item()
    frames=[]
    for seed in (5,6):
        gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
        start=time.perf_counter()
        memento.ht_2d_moments(a,backend='gpu',random_state=seed,**kwargs)
        torch.cuda.synchronize()
        timing=time.perf_counter()-start
        f=memento.get_2d_ht_result(a);valid=np.isfinite(f.corr_pval)
        f['corr_qval']=np.nan
        f.loc[valid,'corr_qval']=multipletests(f.loc[valid,'corr_pval'],method='fdr_bh')[1]
        f.to_csv(out/f'gpu_seed{seed}.csv',index=False);frames.append(f)
        run=dict(seed=seed,seconds=timing,peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
                 finite=int(valid.sum()),fdr_05=int((f.corr_qval<.05).sum()),settings=u['2d_ht']['gpu'])
        report['gpu_runs'].append(run);print(json.dumps(run),flush=True)
    report['gpu_repeat']=compare(*frames)
    # CPU subset spans the empirical baseline correlation ranking.
    correlations=np.array([u['2d_moments'][g]['corr'] for g in groups])
    order=np.argsort(np.nanmean(correlations,axis=0))
    selected=order[np.linspace(0,len(order)-1,min(args.cpu_pairs,len(order))).astype(int)]
    assigned={pairs[i]:['stim'] for i in selected}
    start=time.perf_counter()
    memento.ht_2d_moments(a,backend='cpu',treatment_for_gene=assigned,num_cpus=2,random_state=5,**kwargs)
    report['cpu_subset_seconds']=time.perf_counter()-start
    cpu=memento.get_2d_ht_result(a);cpu.to_csv(out/'cpu_subset.csv',index=False)
    report['gpu_vs_cpu']=compare(cpu,frames[0]);print('CPU comparison',json.dumps(report['gpu_vs_cpu']),flush=True)
    # Matched-size GPU timing provides a direct comparison, without extrapolation.
    torch.cuda.synchronize();start=time.perf_counter()
    memento.ht_2d_moments(a,backend='gpu',treatment_for_gene=assigned,random_state=5,**kwargs)
    torch.cuda.synchronize();report['gpu_subset_seconds']=time.perf_counter()-start
    original=_gpu._regress;errors=[];checked_pairs=[]
    def checked(means,variances,good,covariate,treatment,nc,designs):
        assert variances is None
        got=original(means,variances,good,covariate,treatment,nc,designs)
        ys=means.cpu().numpy();masks=good.cpu().numpy()
        for i,(ts,cs) in enumerate(designs):
            use=masks[i]
            if use.sum()<2:continue
            ref=hypothesis_test._regress_2d(covariate[list(cs)].values[use],treatment[list(ts)].values[use],ys[i,use],nc[use])
            np.testing.assert_allclose(got[i],ref,rtol=2e-7,atol=2e-10,equal_nan=True)
            finite=np.isfinite(ref)&np.isfinite(got[i])
            if finite.any():errors.append(float(abs(np.asarray(ref)[finite]-got[i][finite]).max()))
            checked_pairs.append(i)
        return got
    with patch.object(_gpu,'_regress',checked):
        memento.ht_2d_moments(a,backend='gpu',treatment_for_gene=assigned,random_state=19,**{**kwargs,'num_boot':2000})
    report['same_input_regression']=dict(pairs=len(checked_pairs),max_abs=max(errors,default=0.))
    (out/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
