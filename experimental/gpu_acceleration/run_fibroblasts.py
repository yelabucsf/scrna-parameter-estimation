"""Full-panel CPU/GPU fibroblast tissue comparison, up to ten CPUs.

Read only selected raw-count rows, not the large normalized/scaled layers.
CPU uses ten threads to avoid ten copies of Python/scientific-library memory.
Completed backend runs are checkpointed and reused on restart.
"""
import os
import sys,time,json,gc
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import h5py
import anndata as ad
import numpy as np
import pandas as pd
from joblib import parallel_config
from threadpoolctl import threadpool_limits
import torch
import memento
from run_cell_types import fdr,compare

out=Path('experimental/gpu_acceleration/results_fibroblasts');out.mkdir(exist_ok=True)
path='/mnt/c/Data/ANTXR2_workspace/data/ts_stromal.h5ad'
reference='subcutaneous adipose tissue';target='bladder organ'
start=time.perf_counter()
with h5py.File(path) as f:
    obs=ad.io.read_elem(f['obs'])
    eligible=(obs.cell_type=='fibroblast')&(obs.method=='10X')&obs.tissue.isin([reference,target])
    counts=obs.loc[eligible].groupby(['donor_id','tissue'],observed=True).size().unstack(fill_value=0)
    donors=counts.index[(counts[reference]>=30)&(counts[target]>=30)]
    counts['included']=counts.index.isin(donors);counts.to_csv(out/'donor_cell_counts.csv')
    keep=np.flatnonzero((eligible&obs.donor_id.isin(donors)).values)
    var=ad.io.read_elem(f['raw/var'])
    x=ad.io.sparse_dataset(f['raw/X'])[keep].tocsr()
    assert np.isfinite(x.data).all() and (x.data>=0).all() and (x.data==np.rint(x.data)).all()
    a=ad.AnnData(x,obs=obs.iloc[keep].copy(),var=var)
    del x,obs
print('Loaded',a.shape,'donors',donors.tolist(),flush=True)
a.obs['capture_rate']=.07
memento.setup_memento(a,q_column='capture_rate',min_cell_count=30)
memento.create_groups(a,label_columns=['donor_id','tissue'])
memento.compute_1d_moments(a,min_perc_group=.7)
u=a.uns['memento'];groups=u['groups']
meta=a.obs.drop_duplicates('memento_group').set_index('memento_group').loc[groups]
tx=pd.DataFrame({'bladder_vs_adipose':(meta.tissue==target).astype(float)},index=groups)
cov=pd.get_dummies(meta.donor_id,drop_first=True,dtype=float)
cov=cov.loc[:,cov.any(axis=0)]
report=dict(data=path,count_source='raw/X',cell_type='fibroblast',method='10X',reference=reference,target=target,
    donors=donors.tolist(),cells=a.n_obs,genes=a.n_vars,groups=len(groups),num_boot=10000,q=.07,min_perc_group=.7,
    preparation_seconds=time.perf_counter()-start,cpu_affinity=sorted(os.sched_getaffinity(0)),cpu_workers=10,
    cpu_backend='joblib threading; BLAS threads=1',torch=torch.__version__,gpu=torch.cuda.get_device_name(0),runs=[],
    design='Donor x tissue groups; tissue indicator with donor covariates; complete pairs >=30 cells/tissue',
    group_counts={g:int(u['group_cells'][g].shape[0]) for g in groups})
if (out/'report.json').exists():
    previous=json.loads((out/'report.json').read_text())
    assert all(previous[k]==report[k] for k in ['cells','genes','num_boot','donors'])
    report['runs']=previous['runs']
print('Prepared',json.dumps(report),flush=True)
(out/'report.json').write_text(json.dumps(report,indent=2))
kwargs=dict(treatment=tx,covariate=cov,num_boot=10000,approx='norm',resample_rep=False)
results={}
torch.set_num_threads(min(10,len(os.sched_getaffinity(0))))
torch.ones(1,device='cuda').sum().item()
for backend,seed in [('gpu',5),('gpu',6),('cpu',5)]:
    dest=out/f'{backend}_seed{seed}.csv'
    if dest.exists() and any(r['backend']==backend and r['seed']==seed for r in report['runs']):
        results[(backend,seed)]=pd.read_csv(dest);continue
    gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
    started=time.perf_counter()
    with parallel_config(backend='threading'),threadpool_limits(limits=1):
        memento.ht_1d_moments(a,backend=backend,random_state=seed,num_cpus=10,verbose=5 if backend=='cpu' else 0,**kwargs)
    torch.cuda.synchronize();elapsed=time.perf_counter()-started
    frame=fdr(memento.get_1d_ht_result(a));frame.to_csv(dest,index=False)
    results[(backend,seed)]=frame
    row=dict(backend=backend,seed=seed,seconds=elapsed,genes=len(frame),
        de_finite=int(np.isfinite(frame.de_pval).sum()),dv_finite=int(np.isfinite(frame.dv_pval).sum()),
        de_fdr_05=int((frame.de_qval<.05).sum()),dv_fdr_05=int((frame.dv_qval<.05).sum()))
    if backend=='gpu':row.update(peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,settings=u['1d_ht']['gpu'])
    report['runs'].append(row);(out/'report.json').write_text(json.dumps(report,indent=2))
    print('Completed',json.dumps(row),flush=True)
report['gpu_repeat']=compare(results[('gpu',5)],results[('gpu',6)])
report['gpu_vs_cpu']=compare(results[('cpu',5)],results[('gpu',5)])
for kind in ('de','dv'):
    cpu=results[('cpu',5)].set_index('gene')[kind+'_qval']<.05
    gpu=results[('gpu',5)].set_index('gene')[kind+'_qval']<.05
    report[kind+'_fdr_agreement']=dict(both=int((cpu&gpu).sum()),cpu_only=int((cpu&~gpu).sum()),gpu_only=int((gpu&~cpu).sum()))
(out/'report.json').write_text(json.dumps(report,indent=2))
print('Comparisons',json.dumps({k:report[k] for k in ['gpu_repeat','gpu_vs_cpu','de_fdr_agreement','dv_fdr_agreement']}),flush=True)
