"""Validate and benchmark shared cell weights versus compressed-state bootstrap."""
import os
import argparse
import gc
import json
import pickle
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch
from real_data_bench import np,torch,memento,comparison
from scipy import sparse
from threadpoolctl import threadpool_limits
from memento import bootstrap,estimator
from gpu_1d import gpu_execution
from gpu_1d_memory import sample_packed
from gpu_states import compress_group,GPUStateDispatch
from cell_matmul import coefficients,weight_chunk,moments,matrix_bootstrap,CellDispatch,synchronize


def load():
    with open('experimental/gpu_acceleration/results/prepared.pkl','rb') as f:return pickle.load(f)


def to_np(x):return x.cpu().numpy() if isinstance(x,torch.Tensor) else x


def compressed(x,sf,q,b,device,seed):
    stages={}
    if device=='cpu':
        data=sparse.csc_matrix(x)
        def one(i):
            return bootstrap._bootstrap_1d(data[:,i],sf,q,estimator._hyper_1d_relative,
                                          num_boot=b,rng=np.random.default_rng(seed+i))
        with ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(one,range(x.shape[1])))
        return np.array([r[0] for r in results]),np.array([r[1] for r in results]),stages
    start=time.perf_counter()
    packed=compress_group(x.T,sf,np.full(x.shape[1],q))
    sizes=packed.sizes
    synchronize(device);stages['preparation']=time.perf_counter()-start
    result=torch.empty((2,x.shape[1],b),device='cuda')
    start=time.perf_counter()
    for bucket in np.unique((sizes-1)//64):
        ids=np.flatnonzero((sizes-1)//64==bucket)
        kk=torch.as_tensor(sizes[ids],device='cuda')
        pos=torch.arange(int(sizes[ids].max()),device='cuda')[:,None]
        index=torch.as_tensor(packed.starts[ids],device='cuda')[None,:]+torch.minimum(pos,kk[None,:]-1)
        coeff=torch.where((pos<kk[None,:])[None],packed.coeff[:,index],0.)
        ns=torch.full((len(ids),1),float(len(x)),device='cuda')
        step=bootstrap._get_batch_size(2*len(ids),b,None)
        for lo in range(0,b,step):
            (mu,var),_,_=sample_packed(coeff,ns,min(step,b-lo))
            result[0,ids,lo:lo+step]=mu;result[1,ids,lo:lo+step]=var
    synchronize(device);stages['sampling']=time.perf_counter()-start
    return result[0],result[1],stages


def validate(out):
    report={};rng=np.random.default_rng(91)
    n=7;b=100000
    for device in ('cpu','cuda'):
        w=to_np(weight_chunk(n,b,device,rng))
        np.testing.assert_array_equal(w.sum(1),np.full(b,n))
        assert np.all(w>=0) and np.all(w==np.floor(w))
        expected=np.eye(n)-np.ones((n,n))/n
        np.testing.assert_allclose(w.mean(0),1,atol=.015,rtol=0)
        np.testing.assert_allclose(np.cov(w.T,bias=True),expected,atol=.018,rtol=0)
        report[device+'_weights']={'max_mean_error':float(abs(w.mean(0)-1).max()),
                                   'max_covariance_error':float(abs(np.cov(w.T,bias=True)-expected).max())}
    _,a,_,_=load();u=a.uns['memento']
    groups=sorted(u['groups'],key=lambda g:u['group_cells'][g].shape[0])
    report['real']=[]
    for group in (groups[0],groups[-1]):
        data=u['group_cells'][group]
        means=np.asarray(data.mean(0)).ravel()
        chosen=np.argsort(means)[np.linspace(0,data.shape[1]-1,16).astype(int)]
        x=data[:,chosen].toarray();sf=u['approx_size_factor'][group];q=u['group_q'][group]
        w=weight_chunk(len(x),512,'cpu',rng)
        c64=coefficients(x,sf,q,np.float64)
        refm,refv=moments(w.astype(np.float64),c64,len(x),'cpu')
        # Aggregate the SAME cell counts to each gene's exact compressed states.
        # This isolates estimator equivalence from Monte Carlo differences.
        sf_codes,_=__import__('pandas').factorize(sf,sort=False)
        for i in range(x.shape[1]):
            key=sf_codes*(int(x[:,i].max())+1)+x[:,i].astype(np.int64)
            _,inverse=np.unique(key,return_inverse=True)
            state=bootstrap._unique_expr(sparse.csc_matrix(x[:,i,None]),sf)
            aggregated=w.astype(np.float64)@np.eye(len(state[3]))[inverse]
            m,v=estimator._hyper_1d_relative((state[2],aggregated.T),len(x),q,(state[0],state[1]))
            np.testing.assert_allclose(m,refm[i],rtol=2e-12,atol=1e-12)
            np.testing.assert_allclose(v,refv[i],rtol=2e-11,atol=1e-12)
        checks={}
        for device in ('cpu','cuda'):
            c=coefficients(x,sf,q)
            ww=w
            if device=='cuda':c=torch.as_tensor(c,device='cuda');ww=torch.as_tensor(w,device='cuda')
            m,v=moments(ww,c,len(x),device);m=to_np(m);v=to_np(v)
            np.testing.assert_allclose(m,refm,rtol=2e-5,atol=1e-8)
            scaled=abs(v-refv)/np.maximum(abs(refv)+refm**2,1e-12)
            assert scaled.max()<2e-5
            checks[device+'_max_scaled_variance_error']=float(scaled.max())
        cm,cv,_=compressed(x,sf,q,40000,'cpu',73)
        gm,gv,_=matrix_bootstrap(x,sf,q,40000,'cuda',seed=74)
        ratios=[]
        for left,right in ((cm,to_np(gm)),(cv,to_np(gv))):
            sd=left.std(axis=1);valid=sd>1e-10
            ratio=right.std(axis=1)[valid]/sd[valid]
            np.testing.assert_allclose(ratio,1,rtol=.05)
            ratios.append({'min':float(ratio.min()),'median':float(np.median(ratio)),'max':float(ratio.max())})
        report['real'].append({'group':group,'cells':len(x),'same_weight_checks':checks,'sd_ratios_mean_var':ratios})
    (out/'validation.json').write_text(json.dumps(report,indent=2))
    print('Validation passed',json.dumps(report),flush=True)


def sweep(args,out):
    spec,a,_,_=load();u=a.uns['memento']
    # Pooled cells provide computational workloads only, not biological tests.
    x=sparse.vstack([u['group_cells'][g] for g in u['groups']]).toarray()
    sf=np.concatenate([u['approx_size_factor'][g] for g in u['groups']])
    order=np.argsort(x.mean(0));rng=np.random.default_rng(106)
    report={'args':vars(args),'dataset':spec,'cpu_affinity':sorted(os.sched_getaffinity(0)),
            'cpu_threads':2,'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,
            'tf32':torch.backends.cuda.matmul.allow_tf32,'runs':[],
            'scope':'Computational pool of real CD14+ cells; no biological inference on pooled/resampled groups. Above 5341 cells, sampling with replacement. Preparation and sampling included; regression excluded.'}
    modes=args.sweep_modes
    # Warm libraries and device before measured calls.
    matrix_bootstrap(x[:100,:16],sf[:100],.07,128,'cpu')
    matrix_bootstrap(x[:100,:16],sf[:100],.07,128,'cuda')
    compressed(x[:100,:16],sf[:100],.07,128,'cuda',1)
    for genes in args.gene_counts:
        ids=(order[[len(order)//2]] if genes==1 else
             order[np.linspace(0,len(order)-1,min(genes,len(order))).astype(int)])
        for n in args.cells:
            indices=rng.choice(len(x),n,replace=n>len(x))
            xx=x[indices][:,ids];ss=sf[indices]
            sizes=[len(bootstrap._unique_expr(sparse.csc_matrix(xx[:,i,None]),ss)[3]) for i in range(len(ids))]
            for repeat in range(args.repeats):
                for mode in (modes if repeat%2==0 else modes[::-1]):
                    device='cuda' if mode.startswith('gpu') else 'cpu'
                    gc.collect();torch.cuda.empty_cache();torch.manual_seed(107+repeat)
                    torch.cuda.reset_peak_memory_stats()
                    synchronize(device);start=time.perf_counter()
                    if mode.endswith('compressed'):m,v,stages=compressed(xx,ss,.07,args.boots,device,107+repeat)
                    else:m,v,stages=matrix_bootstrap(xx,ss,.07,args.boots,device,seed=107+repeat,
                                                   dtype=np.float64 if mode=='cpu_matrix64' else np.float32)
                    synchronize(device);elapsed=time.perf_counter()-start
                    m=to_np(m);v=to_np(v)
                    assert np.isfinite(m).all() and np.isfinite(v).all()
                    row={'genes':len(ids),'cells':n,'boots':args.boots,'repeat':repeat,'mode':mode,
                         'seconds':elapsed,'stages':stages,'median_states':float(np.median(sizes)),
                         'max_states':max(sizes),'peak_gpu_allocated_gib':torch.cuda.max_memory_allocated()/1024**3,
                         'median_mean_sd':float(np.median(m.std(1))), 'median_variance_sd':float(np.median(v.std(1)))}
                    report['runs'].append(row)
                    (out/'sweep.json').write_text(json.dumps(report,indent=2))
                    print(json.dumps(row),flush=True)
                    del m,v


def full(args,out):
    spec,a,treatment,covariate=load();u=a.uns['memento']
    order=np.argsort(np.mean([u['1d_moments'][g][0] for g in u['groups']],axis=0))
    take=order[np.linspace(0,len(order)-1,min(args.genes,len(order))).astype(int)]
    genes=a.var_names[take].tolist()
    report={'args':vars(args),'dataset':spec,'genes':len(genes),'runs':[],
            'cpu_affinity':sorted(os.sched_getaffinity(0)),'cpu_threads':2,
            'torch':torch.__version__,'gpu':torch.cuda.get_device_name(0),'tf32':torch.backends.cuda.matmul.allow_tf32}
    outputs={}
    for trial,mode in enumerate(args.modes):
        if mode=='gpu_compressed':dispatch=GPUStateDispatch(args.batch,1024,args.validate)
        elif mode in ('gpu_matrix','cpu_matrix'):
            dispatch=CellDispatch(args.batch,'cuda' if mode=='gpu_matrix' else 'cpu',args.validate,seed=45+trial)
        else:raise ValueError(mode)
        gc.collect();torch.cuda.empty_cache();torch.manual_seed(45+trial)
        torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
        with patch.object(bootstrap,'_DEFAULT_BOOTSTRAP_BATCH_BYTES',256*1024**2),gpu_execution(dispatch):
            memento.ht_1d_moments(a,treatment=treatment,covariate=covariate,
                treatment_for_gene={g:['stim'] for g in genes},num_boot=args.boots,
                num_cpus=1,verbose=0,random_state=5)
        torch.cuda.synchronize();elapsed=time.perf_counter()-start
        result={k:np.asarray(u['1d_ht'][k]).copy() for k in ('mean_coef','mean_se','mean_asl','var_coef','var_se','var_asl')}
        assert all(np.isfinite(v).all() for v in result.values())
        row={'mode':mode,'trial':trial,'seconds':elapsed,'stages':dispatch.timings,
             'checks':dispatch.checks,'peak_gpu_allocated_gib':torch.cuda.max_memory_allocated()/1024**3}
        if 'gpu_compressed' in outputs:row['vs_compressed']=comparison(outputs['gpu_compressed'],result)
        if mode in outputs:row['vs_same_mode_repeat']=comparison(outputs[mode],result)
        outputs[mode]=result
        np.savez(out/f'{trial}_{mode}.npz',genes=np.array(genes),**result)
        report['runs'].append(row);(out/'full.json').write_text(json.dumps(report,indent=2))
        print(json.dumps({k:v for k,v in row.items() if k not in ('vs_compressed','vs_same_mode_repeat')}),flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--phase',choices=['validate','sweep','full'],required=True)
    p.add_argument('--out',default='experimental/gpu_acceleration/results_cell_matmul')
    p.add_argument('--boots',type=int,default=10000)
    p.add_argument('--genes',type=int,default=1742)
    p.add_argument('--batch',type=int,default=256)
    p.add_argument('--gene-counts',type=int,nargs='+',default=[128])
    p.add_argument('--cells',type=int,nargs='+',default=[100,200,500,1000,2000,5000,10000,20000])
    p.add_argument('--repeats',type=int,default=2)
    p.add_argument('--modes',nargs='+',default=['gpu_compressed','gpu_matrix','cpu_matrix','gpu_matrix','gpu_compressed'])
    p.add_argument('--sweep-modes',nargs='+',choices=['cpu_compressed','cpu_matrix','cpu_matrix64','gpu_compressed','gpu_matrix'],
                   default=['cpu_compressed','cpu_matrix','gpu_compressed','gpu_matrix'])
    p.add_argument('--validate',action='store_true')
    args=p.parse_args()
    if min(args.boots,args.genes,args.batch,args.repeats,*args.gene_counts,*args.cells)<1:p.error('Positive counts required')
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    with threadpool_limits(limits=2):
        if args.phase=='validate':validate(out)
        elif args.phase=='sweep':sweep(args,out)
        else:full(args,out)


if __name__=='__main__':main()
