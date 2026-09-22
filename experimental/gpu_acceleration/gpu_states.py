"""Deterministic batched GPU state preparation; no new sampling algorithm."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
import torch
from gpu_1d_memory import WideGPUDispatch, sample_packed


@dataclass
class CompressedGroup:
    coeff: torch.Tensor
    expr: torch.Tensor
    counts: torch.Tensor
    inv_sf: torch.Tensor
    inv_sf_sq: torch.Tensor
    sizes: np.ndarray
    starts: np.ndarray
    n: int


def compress_group(dense, size_factor, qs):
    """Compress [genes, cells] in CPU-reference state order using integer keys.

    Size-factor coding is done once per group on CPU to preserve first-seen
    order. Counting, decoding and coefficient preparation run on GPU. Only
    per-gene sizes/offsets return to CPU for scheduling.
    """
    dense = np.asarray(dense)
    sf = np.asarray(size_factor)
    if dense.ndim != 2 or min(dense.shape) < 1:
        raise ValueError('dense must be a nonempty genes-by-cells array')
    g,n = dense.shape
    if n > 2**24:
        raise ValueError('float32 sampler counts require n <= 2**24')
    if sf.ndim != 1 or sf.size != n or np.any(~np.isfinite(sf)) or np.any(sf<=0):
        raise ValueError('one finite positive size factor per cell is required')
    if np.any(~np.isfinite(dense)) or np.any(dense<0) or np.any(dense != np.rint(dense)):
        raise ValueError('GPU state preparation requires finite nonnegative integer counts')
    qs = np.asarray(qs,dtype=np.float64)
    if qs.shape != (g,) or np.any(~np.isfinite(qs)) or np.any(qs<0) or np.any(qs>=1):
        raise ValueError('one capture rate in [0,1) per gene is required')
    codes, values = pd.factorize(sf,sort=False)
    radix = int(dense.max())+1
    state_space = len(values)*radix
    if g*state_space > np.iinfo(np.int64).max:
        raise ValueError('integer state keys would overflow int64')
    x = torch.as_tensor(dense.astype(np.int64),device='cuda')
    sf_code = torch.as_tensor(codes,device='cuda')
    gene = torch.arange(g,device='cuda')[:,None]
    keys = gene*state_space + sf_code[None,:]*radix + x
    unique,counts = torch.unique(keys.flatten(),sorted=True,return_counts=True)
    row = unique//state_space
    code = unique % state_space
    expression = code % radix
    sf_index = code//radix
    sizes = torch.bincount(row,minlength=g).cpu().numpy()
    starts = np.r_[0,np.cumsum(sizes)[:-1]]
    value_array = np.asarray(values)
    if not np.issubdtype(value_array.dtype,np.floating):
        value_array = value_array.astype(np.float64)
    sf_values = torch.as_tensor(value_array,device='cuda')
    # Preserve the reference's size-factor dtype for inverse and square,
    # then promote to fp64 exactly as pack_states does.
    inv = 1/sf_values[sf_index]
    inv2 = inv.square()
    e = expression.double()
    q = torch.as_tensor(qs,device='cuda')[row]
    # All genes have n cells. Previous genes contribute exactly row*n counts.
    remaining = n-(counts.cumsum(0)-counts-row*n)
    cond = (counts.double()/remaining.double()).clamp(0,1)
    c1 = e*inv.double()
    c2 = (e.square()-(1-q)*e)*inv2.double()
    coeff = torch.stack((cond,c1,c2)).float()
    return CompressedGroup(coeff,expression,counts,inv,inv2,sizes,starts,n)


class GPUStateDispatch(WideGPUDispatch):
    """Prepare each gene chunk on GPU, preserving cross-group scheduling."""
    def __init__(self,batch_genes=256,row_batch=1024,validate=False):
        super().__init__(batch_genes,row_batch,False,validate)
        self.coefficients = None

    def make_buckets(self,tasks):
        compressed=[]
        offsets=[]
        offset=0
        for group in range(len(tasks[0]['cells'])):
            sf=tasks[0]['approx_sf'][group]
            if any(t['approx_sf'][group] is not sf and not np.array_equal(t['approx_sf'][group],sf)
                   for t in tasks):
                raise ValueError('GPU preparation requires common group size factors')
            dense=np.concatenate([t['cells'][group].toarray() for t in tasks],axis=1).T
            packed=compress_group(dense,sf,[t['q'][group] for t in tasks])
            offsets.append(offset)
            offset+=packed.coeff.shape[1]
            compressed.append(packed)
        self.coefficients=torch.cat([p.coeff for p in compressed],dim=1)
        buckets={}
        # Same insertion order and stable k-sort as the CPU preparation path.
        for i,t in enumerate(tasks):
            for group,p in enumerate(compressed):
                mu,rv=t['true_mean'][group],t['true_res_var'][group]
                k=int(p.sizes[i])
                if np.isnan(mu) or np.isnan(rv) or mu==0 or rv<0 or k<=1:
                    continue
                state=(int(p.starts[i])+offsets[group],k,p.n)
                buckets.setdefault((k-1)//64,[]).append((i,group,state))
        return buckets

    @staticmethod
    def state_count(state):
        return state[1]

    def sample_rows(self,states,qs,boots):
        starts,k,ns=map(np.asarray,zip(*states))
        kk=torch.as_tensor(k,device='cuda')
        pos=torch.arange(int(k.max()),device='cuda')[:,None]
        valid=pos<kk[None,:]
        idx=torch.as_tensor(starts,device='cuda')[None,:]+torch.minimum(pos,kk[None,:]-1)
        coeff=torch.where(valid[None,:,:],self.coefficients[:,idx],0.)
        ns_t=torch.as_tensor(ns,device='cuda',dtype=torch.float32)[:,None]
        result,remaining,_=sample_packed(coeff,ns_t,boots)
        self.sampler.calls+=1
        self.sampler.last_remaining=remaining
        return result
