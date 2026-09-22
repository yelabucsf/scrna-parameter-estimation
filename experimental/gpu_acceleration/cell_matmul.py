"""Experimental shared cell bootstrap, retaining the existing 1D estimator.

Uniform cell draws are counted in bounded chunks, then reused across genes.
No package backend/default changes. Uses the same fixed approximate size factors
as the compressed bootstrap. CUDA TF32 must be disabled by the runner.
"""
import time
import numpy as np
import torch
from gpu_1d import GPUDispatch, fill_positive, regress
from memento.hypothesis_test import _fill


def synchronize(device):
    if device=='cuda': torch.cuda.synchronize()


def coefficients(x,sf,q,dtype=np.float32):
    """Cell x (mean coefficients | second moment coefficients)."""
    x=np.asarray(x,dtype=np.float64)
    inv=1/np.asarray(sf)
    inv2=inv**2
    first=x*inv[:,None]
    second=(x*x-(1-np.asarray(q))*x)*inv2[:,None]
    return np.ascontiguousarray(np.concatenate((first,second),axis=1),dtype=dtype)


def weight_chunk(n,b,device,rng=None):
    if not 0<n<=2**24: raise ValueError('Require 0 < cells <= 2**24')
    if device=='cuda':
        indices=torch.randint(n,(b,n),device='cuda')
        weights=torch.zeros((b,n),device='cuda',dtype=torch.float32)
        weights.scatter_add_(1,indices,torch.ones((),device='cuda').expand_as(weights))
        return weights
    if rng is None: raise ValueError('CPU sampling requires an explicit generator')
    indices=rng.integers(n,size=(b,n),dtype=np.int64)
    indices+=np.arange(b,dtype=np.int64)[:,None]*n
    return np.bincount(indices.ravel(),minlength=b*n).reshape(b,n).astype(np.float32)


def moments(weights,coef,n,device):
    product=weights@coef
    g=coef.shape[1]//2
    mean=product[:,:g].T/n
    second=product[:,g:].T/n
    return mean,second-mean*mean


def fill_cpu(values,rng):
    """Use memento's CPU replacement routine instead of GPU searchsorted."""
    valid=[]
    for row in values.numpy():
        valid.append(_fill(row,rng=rng) is not None)
    return values,torch.tensor(valid,dtype=torch.bool)


def matrix_bootstrap(x,sf,q,boots,device,chunk=512,seed=5,dtype=np.float32):
    """One group microbenchmark, including preparation and weight generation."""
    stages=dict(preparation=0.,weights=0.,multiply=0.)
    synchronize(device);start=time.perf_counter()
    c=coefficients(x,sf,q,dtype)
    if device=='cuda': c=torch.as_tensor(c,device=device)
    synchronize(device);stages['preparation']=time.perf_counter()-start
    rng=np.random.default_rng(seed)
    means=[];variances=[]
    for lo in range(0,boots,chunk):
        synchronize(device);start=time.perf_counter()
        w=weight_chunk(len(x),min(chunk,boots-lo),device,rng)
        if device=='cpu' and c.dtype!=w.dtype:w=w.astype(c.dtype)
        synchronize(device);stages['weights']+=time.perf_counter()-start
        start=time.perf_counter()
        m,v=moments(w,c,len(x),device)
        means.append(m);variances.append(v)
        synchronize(device);stages['multiply']+=time.perf_counter()-start
    cat=(lambda values:torch.cat(values,dim=1)) if device=='cuda' else (lambda values:np.concatenate(values,axis=1))
    return cat(means),cat(variances),stages


class CellDispatch(GPUDispatch):
    """Share each biological group's cell weights across every gene chunk."""
    def __init__(self,batch_genes=256,device='cuda',validate=False,seed=5,weight_chunk_size=512):
        super().__init__(batch_genes,validate=validate)
        self.device=device
        self.rng=np.random.default_rng(seed)
        self.fill_rng=np.random.default_rng(seed+100000)
        self.weight_chunk_size=weight_chunk_size
        self.weights={}
        self.timings=dict(preparation=0.,weights=0.,bootstrap=0.,transform=0.,regression=0.)

    def __call__(self,jobs):
        self.weights.clear()
        try: return super().__call__(jobs)
        finally: self.weights.clear()

    def group_weights(self,group,n,b):
        if group in self.weights: return self.weights[group]
        if self.device=='cuda': w=torch.empty((b,n),device='cuda')
        else: w=np.empty((b,n),dtype=np.float32)
        for lo in range(0,b,self.weight_chunk_size):
            w[lo:lo+self.weight_chunk_size]=weight_chunk(n,min(self.weight_chunk_size,b-lo),self.device,self.rng)
        self.weights[group]=w
        return w

    def run_chunk(self,tasks):
        device=self.device;g=len(tasks);h=len(tasks[0]['cells']);b=tasks[0]['num_boot']
        means=torch.full((g,h,b+1),torch.nan,device=device,dtype=torch.float64)
        variances=torch.full_like(means,torch.nan)
        good=torch.zeros((g,h),device=device,dtype=torch.bool)
        for group in range(h):
            synchronize(device);start=time.perf_counter()
            sf=tasks[0]['approx_sf'][group]
            if any(not np.array_equal(t['approx_sf'][group],sf) for t in tasks):
                raise ValueError('Common group size factors required')
            x=np.concatenate([t['cells'][group].toarray() for t in tasks],axis=1)
            q=np.array([t['q'][group] for t in tasks])
            coef=coefficients(x,sf,q)
            if device=='cuda':coef=torch.as_tensor(coef,device=device)
            eligible=np.array([not(np.isnan(t['true_mean'][group]) or np.isnan(t['true_res_var'][group])
                or t['true_mean'][group]==0 or t['true_res_var'][group]<0) for t in tasks])
            if np.all(sf==sf[0]):eligible &= np.any(x!=x[:1],axis=0)
            synchronize(device);self.timings['preparation']+=time.perf_counter()-start
            start=time.perf_counter()
            w=self.group_weights(group,len(x),b)
            synchronize(device);self.timings['weights']+=time.perf_counter()-start
            start=time.perf_counter()
            mu,v=moments(w,coef,len(x),device)
            mu=torch.as_tensor(mu,device=device).double().contiguous()
            v=torch.as_tensor(v,device=device).double().contiguous()
            synchronize(device);self.timings['bootstrap']+=time.perf_counter()-start
            start=time.perf_counter()
            fit=torch.as_tensor(np.array([t['mv_fit'][group] for t in tasks]),device=device,dtype=torch.float64)
            lm=mu.log()
            rv=torch.exp(v.log()-(fit[:,0,None]*lm+fit[:,1,None])*lm-fit[:,2,None])
            rv=torch.where((mu>0)&(v>0),rv,torch.nan)
            if device=='cpu':
                mu,valid_m=fill_cpu(mu,self.fill_rng)
                rv,valid_v=fill_cpu(rv,self.fill_rng)
            else:
                mu,valid_m=fill_positive(mu);rv,valid_v=fill_positive(rv)
            means[:,group,1:]=mu.log();variances[:,group,1:]=rv.log()
            means[:,group,0]=torch.as_tensor(np.log([t['true_mean'][group] for t in tasks]),device=device)
            variances[:,group,0]=torch.as_tensor(np.log([t['true_res_var'][group] for t in tasks]),device=device)
            good[:,group]=torch.as_tensor(eligible,device=device)&valid_m&valid_v
            synchronize(device);self.timings['transform']+=time.perf_counter()-start
        start=time.perf_counter()
        result=regress(means,variances,good,tasks,device=device)
        synchronize(device);self.timings['regression']+=time.perf_counter()-start
        if self.validate:
            reference=np.asarray(self.cpu_regress(means,variances,good,tasks))
            np.testing.assert_allclose(result,reference,rtol=2e-7,atol=2e-10,equal_nan=True)
            other=regress(means,variances,good,tasks,device='cpu' if device=='cuda' else 'cuda')
            np.testing.assert_allclose(other,reference,rtol=2e-7,atol=2e-10,equal_nan=True)
            self.checks.append({'reference_cpu_max_abs':float(np.nanmax(abs(np.asarray(result)-reference))),
                               'other_device_max_abs':float(np.nanmax(abs(np.asarray(other)-reference)))})
        return result
