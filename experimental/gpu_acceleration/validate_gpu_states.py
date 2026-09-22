"""Exact CPU/GPU compression checks, before any sampler changes."""
import os
import argparse
import json
import pickle
from pathlib import Path
from real_data_bench import np, torch
from scipy import sparse
from memento import bootstrap
from gpu_1d_memory import pack_states, sample_packed, MixedSampler
from gpu_states import compress_group


def check_group(dense,sf,qs):
    packed=compress_group(dense,sf,qs)
    expr,counts,inv,inv2,coeff=[x.cpu().numpy() for x in
        (packed.expr,packed.counts,packed.inv_sf,packed.inv_sf_sq,packed.coeff)]
    nonidentical=0
    max_abs=0.
    for i in range(len(dense)):
        state=bootstrap._unique_expr(sparse.csc_matrix(dense[i,:,None]),sf)
        lo=int(packed.starts[i]); hi=lo+int(packed.sizes[i])
        np.testing.assert_array_equal(expr[lo:hi],state[2][:,0])
        np.testing.assert_array_equal(counts[lo:hi],state[3])
        np.testing.assert_allclose(inv[lo:hi],state[0][:,0],rtol=2e-7,atol=0)
        np.testing.assert_allclose(inv2[lo:hi],state[1][:,0],rtol=3e-7,atol=0)
        ref,_=pack_states([state],[qs[i]])
        target=ref[:,:,0]
        np.testing.assert_array_equal(coeff[0,lo:hi],target[0])
        np.testing.assert_allclose(coeff[:,lo:hi],target,rtol=3e-7,atol=1e-12)
        nonidentical+=int(np.count_nonzero(coeff[:,lo:hi]!=target))
        max_abs=max(max_abs,float(np.max(abs(coeff[:,lo:hi]-target))))
        assert counts[lo:hi].sum()==dense.shape[1]
    return {'genes':len(dense),'cells':dense.shape[1],'states':int(len(counts)),
            'nonidentical_coefficients':nonidentical,'max_abs_coefficient_error':max_abs}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--prepared',default='experimental/gpu_acceleration/results/prepared.pkl')
    p.add_argument('--out',default='experimental/gpu_acceleration/results_gpu_states/validation.json')
    args=p.parse_args()
    rng=np.random.default_rng(707)
    report={'synthetic':[],'real_groups':[]}
    for dtype in (np.float32,np.float64,np.int64):
        for n in (1,17,100):
            dense=rng.poisson(2,size=(7,n))
            dense[0]=0
            dense[1]=5
            sf=rng.choice([1,2,3],n).astype(dtype)
            report['synthetic'].append(check_group(dense,sf,np.linspace(0,.4,len(dense))))
    # Force the CPU reference's sparse state-space fallback and test noninteger SFs.
    dense=np.array([[0,100000000,0,100000000],[1,1,1,1],[0,0,0,0]])
    report['synthetic'].append(check_group(dense,np.array([1.25,.75,.75,1.25]),np.full(3,.07)))
    invalid=[(np.array([[.5,1]]),np.ones(2)),(np.array([[-1,1]]),np.ones(2)),
             (np.array([[0,np.nan]]),np.ones(2)),(np.array([[0,1]]),np.array([0.,1.])),
             (np.array([[0,1]]),np.array([np.inf,1.])),
             (np.array([[0,1.e19]]),np.ones(2))]
    for x,sf in invalid:
        try: compress_group(x,sf,np.array([.07]))
        except ValueError: pass
        else: raise AssertionError('Invalid input accepted')
    report['rejected_invalid_cases']=len(invalid)
    with open(args.prepared,'rb') as f: _,a,_,_=pickle.load(f)
    u=a.uns['memento']
    for group in u['groups']:
        dense=u['group_cells'][group].toarray().T
        result=check_group(dense,u['approx_size_factor'][group],np.full(len(dense),u['group_q'][group]))
        result['group']=group
        report['real_groups'].append(result)
        print('Checked',group,result,flush=True)
    # Identical RNG and coefficient rows: isolate preparation from MC variation.
    group=u['groups'][0]
    dense=u['group_cells'][group][:,:8].toarray().T
    sf=u['approx_size_factor'][group]; qs=np.full(8,u['group_q'][group])
    packed=compress_group(dense,sf,qs)
    cpu_states=[bootstrap._unique_expr(sparse.csc_matrix(x[:,None]),sf) for x in dense]
    k=int(packed.sizes.max())
    idx=torch.as_tensor(packed.starts,device='cuda')[None,:]+torch.minimum(
        torch.arange(k,device='cuda')[:,None],torch.as_tensor(packed.sizes,device='cuda')[None,:]-1)
    valid=torch.arange(k,device='cuda')[:,None]<torch.as_tensor(packed.sizes,device='cuda')[None,:]
    coeff=torch.where(valid[None],packed.coeff[:,idx],0.)
    n=torch.full((len(dense),1),float(dense.shape[1]),device='cuda')
    torch.manual_seed(414)
    reference=MixedSampler()(cpu_states,qs,2000)
    torch.manual_seed(414)
    result,remaining,_=sample_packed(coeff,n,2000)
    assert bool((remaining==0).all())
    errors=[]
    for ref,got in zip(reference,result):
        np.testing.assert_allclose(got.cpu().numpy(),ref.cpu().numpy(),rtol=2e-5,atol=1e-12)
        errors.append(float((got-ref).abs().max()))
    report['same_rng_moment_max_abs_errors']=errors
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2))
    print('All state-preparation checks passed; same-RNG moment errors:',errors,flush=True)


if __name__=='__main__': main()
