"""Real-data paired ctrl/stim tests using the public optional GPU backend."""
import os
import argparse
import gc
import json
import re
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


def fdr(frame):
    frame=frame.copy()
    for kind in ('de','dv'):
        values=frame[kind+'_pval'].to_numpy()
        valid=np.isfinite(values)
        frame[kind+'_qval']=np.nan
        if valid.any():frame.loc[valid,kind+'_qval']=multipletests(values[valid],method='fdr_bh')[1]
    return frame


def compare(reference,other):
    joined=reference.set_index(['gene','tx']).join(other.set_index(['gene','tx']),lsuffix='_ref',rsuffix='_other',how='inner')
    out={}
    for kind in ('de','dv'):
        ref=joined[kind+'_se_ref'].to_numpy();got=joined[kind+'_se_other'].to_numpy()
        valid=np.isfinite(ref)&np.isfinite(got)&(ref>0)
        ratios=got[valid]/ref[valid]
        difference=abs(joined[kind+'_coef_ref'].to_numpy()[valid]-joined[kind+'_coef_other'].to_numpy()[valid])/ref[valid]
        out[kind]={'finite_comparisons':int(valid.sum()),
                   'se_ratio_quantiles':np.quantile(ratios,[0,.05,.5,.95,1]).tolist() if len(ratios) else [],
                   'coefficient_difference_over_se':np.quantile(difference,[.5,.95,1]).tolist() if len(difference) else [],
                   'nan_se_mismatch':int(np.count_nonzero(np.isnan(ref)!=np.isnan(got)))}
    return out


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',default='/mnt/c/Data/memento_workspace/interferon_filtered.h5ad')
    p.add_argument('--out',default='experimental/gpu_acceleration/results_cell_types')
    p.add_argument('--boots',type=int,default=10000)
    p.add_argument('--cpu-genes',type=int,default=16)
    p.add_argument('--cell-types',nargs='+')
    args=p.parse_args()
    if args.boots<2 or args.cpu_genes<1:p.error('boots >=2 and cpu-genes >=1 required')
    torch.set_num_threads(2)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    source=anndata.read_h5ad(args.data,backed='r')
    cell_types=args.cell_types or ['CD14+ Monocytes']+sorted(set(source.obs.cell)-{'CD14+ Monocytes'})
    report={'data':args.data,'boots':args.boots,'seed':5,'q':.07,'min_cells_per_condition':10,
            'design':'Within each cell type: donor x cell-type x condition groups; stim vs ctrl with donor fixed effects; complete donor pairs only.',
            'fdr':'Benjamini-Hochberg separately within each cell type and each moment family',
            'cpu_affinity':sorted(os.sched_getaffinity(0)),'torch':torch.__version__,
            'gpu':torch.cuda.get_device_name(0),'cell_types':[]}
    combined=[];summary=[];all_counts=[]
    torch.ones(1,device='cuda').sum().item()
    try:
        for cell_type in cell_types:
            slug=re.sub(r'[^a-z0-9]+','_',cell_type.lower()).strip('_')
            counts=source.obs.loc[source.obs.cell==cell_type].groupby(['ind','stim'],observed=True).size().unstack(fill_value=0)
            paired=counts.index[(counts.get('ctrl',0)>=10)&(counts.get('stim',0)>=10)]
            record={'cell_type':cell_type,'raw_cells':int((source.obs.cell==cell_type).sum()),
                    'included_donors':[str(v) for v in paired],
                    'excluded_donors':[str(v) for v in counts.index if v not in paired]}
            count_frame=counts.reset_index();count_frame.insert(0,'cell_type',cell_type)
            count_frame['included']=count_frame.ind.isin(paired);all_counts.append(count_frame)
            pd.concat(all_counts).to_csv(out/'donor_cell_counts.csv',index=False)
            if len(paired)<2:
                record['skipped']='Fewer than two complete donor pairs with >=10 cells per condition'
                report['cell_types'].append(record)
                (out/'report.json').write_text(json.dumps(report,indent=2));continue
            start=time.perf_counter()
            keep=(source.obs.cell==cell_type)&source.obs.ind.isin(paired)&source.obs.stim.isin(['ctrl','stim'])
            a=source[keep].to_memory();a.X=sparse.csr_matrix(a.X)
            a.obs['capture_rate']=.07
            memento.setup_memento(a,q_column='capture_rate',min_cell_count=10)
            memento.create_groups(a,label_columns=['ind','cell','stim'])
            memento.compute_1d_moments(a,min_perc_group=.7)
            u=a.uns['memento'];groups=u['groups']
            metadata=a.obs.drop_duplicates('memento_group').set_index('memento_group').loc[groups]
            treatment=pd.DataFrame({'stim':(metadata.stim=='stim').astype(float)},index=groups)
            covariate=pd.get_dummies(metadata.ind,drop_first=True,dtype=float)
            if not len(covariate.columns):covariate=pd.DataFrame({'intercept':np.ones(len(groups))},index=groups)
            record.update(cells=a.n_obs,genes=a.n_vars,groups=len(groups),preparation_seconds=time.perf_counter()-start,
                          group_counts={g:int(u['group_cells'][g].shape[0]) for g in groups})
            kwargs=dict(treatment=treatment,covariate=covariate,num_boot=args.boots,verbose=0,approx='norm',resample_rep=False)
            gpu_results=[];record['gpu_runs']=[]
            for seed in (5,6):
                gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
                start=time.perf_counter()
                memento.ht_1d_moments(a,backend='gpu',random_state=seed,**kwargs)
                torch.cuda.synchronize();elapsed=time.perf_counter()-start
                result=fdr(memento.get_1d_ht_result(a))
                result.insert(0,'cell_type',cell_type)
                result.to_csv(out/f'{slug}_seed{seed}.csv',index=False)
                gpu_results.append(result)
                record['gpu_runs'].append({'seed':seed,'seconds':elapsed,
                    'peak_allocated_gib':torch.cuda.max_memory_allocated()/1024**3,'settings':u['1d_ht']['gpu']})
                print(json.dumps({'cell_type':cell_type,'seed':seed,'cells':a.n_obs,'genes':a.n_vars,'seconds':elapsed}),flush=True)
            primary=gpu_results[0];combined.append(primary)
            pd.concat(combined,ignore_index=True).to_csv(out/'ctrl_vs_stim_1d.csv',index=False)
            record['gpu_repeat']=compare(primary,gpu_results[1])
            order=np.argsort(np.mean([u['1d_moments'][g][0] for g in groups],axis=0))
            chosen=a.var_names[order[np.linspace(0,len(order)-1,min(args.cpu_genes,len(order))).astype(int)]].tolist()
            assigned={g:['stim'] for g in chosen}
            start=time.perf_counter()
            memento.ht_1d_moments(a,backend='cpu',treatment_for_gene=assigned,num_cpus=2,random_state=5,**kwargs)
            record['cpu_subset_seconds']=time.perf_counter()-start
            cpu=memento.get_1d_ht_result(a)
            cpu.to_csv(out/f'{slug}_cpu_subset.csv',index=False)
            record['cpu_subset_genes']=len(chosen);record['gpu_vs_cpu_subset']=compare(cpu,primary)
            # Arithmetic check: apply original CPU regressions to the exact
            # log-bootstrap inputs produced by the public GPU path. Not timed.
            original=_gpu._regress;errors=[]
            def checked(means,variances,good,cov,tx,nc,designs):
                result=original(means,variances,good,cov,tx,nc,designs)
                mm,vv,masks=[value.cpu().numpy() for value in (means,variances,good)]
                for i,(ts,cs) in enumerate(designs):
                    mask=masks[i]
                    if mask.sum()<2:
                        assert np.isnan(result[i]).all();continue
                    ref=hypothesis_test._regress_1d(cov[list(cs)].values[mask].astype(float),
                        tx[list(ts)].values[mask].astype(float),mm[i,mask],vv[i,mask],nc[mask])
                    np.testing.assert_allclose(result[i],ref,rtol=2e-7,atol=2e-10,equal_nan=True)
                    finite=np.isfinite(ref)&np.isfinite(result[i])
                    if finite.any():errors.append(float(abs(np.asarray(result[i])[finite]-np.asarray(ref)[finite]).max()))
                return result
            validation_kwargs={**kwargs,'num_boot':2000}
            with patch.object(_gpu,'_regress',checked):
                memento.ht_1d_moments(a,backend='gpu',treatment_for_gene=assigned,random_state=19,**validation_kwargs)
            record['same_input_cpu_regression_max_abs']=max(errors,default=0.)
            row={'cell_type':cell_type,'cells':a.n_obs,'donor_pairs':len(paired),'genes':a.n_vars,
                 'gpu_seconds':record['gpu_runs'][0]['seconds'],'gpu_repeat_seconds':record['gpu_runs'][1]['seconds'],
                 'de_fdr_05':int((primary.de_qval<.05).sum()),'dv_fdr_05':int((primary.dv_qval<.05).sum()),
                 'de_nonfinite':int((~np.isfinite(primary.de_pval)).sum()),'dv_nonfinite':int((~np.isfinite(primary.dv_pval)).sum())}
            record['summary']=row
            record['top_de']=primary.sort_values(['de_qval','de_pval']).head(8)[['gene','de_coef','de_pval','de_qval']].to_dict('records')
            summary.append(row);report['cell_types'].append(record)
            pd.DataFrame(summary).to_csv(out/'summary.csv',index=False)
            (out/'report.json').write_text(json.dumps(report,indent=2))
            print('Completed',json.dumps(row),flush=True)
            del a,u,primary,gpu_results
    finally:source.file.close()


if __name__=='__main__':main()
