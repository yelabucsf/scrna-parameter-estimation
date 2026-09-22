"""Distributional, padding, and identical-input regression checks on CUDA."""
import os
import json
import argparse
import pickle
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from gpu_1d import sample_states, regress
from memento import hypothesis_test as ht, bootstrap, estimator

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.backends.cuda.matmul.allow_tf32 = False


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--prepared', help='Local prepared.pkl from real_data_bench.py')
    parser.add_argument('--out', help='Write validation JSON')
    args_cli=parser.parse_args()
    torch.manual_seed(73)
    rng = np.random.default_rng(19)
    n, b, q = 100, 100000, .07
    states=[]
    for counts in ([10,30,60], [100], [1,1,3,5,20,70]):
        counts = np.array(counts)
        e=np.arange(len(counts),dtype=float)[:,None]
        inv=np.linspace(.5,2,len(counts))[:,None]
        states.append((inv,inv**2,e,counts))
    mu,var,w=sample_states(states,n,q,b,return_weights=True)
    mu,var,w=[x.double().cpu().numpy() for x in (mu,var,w)]
    np.testing.assert_array_equal(w.sum(0),n)
    report={'sampler':[]}
    for i,(inv,inv2,e,counts) in enumerate(states):
        k=len(counts); x=w[:k,i]; p=counts/n
        expected=n*(np.diag(p)-np.outer(p,p))
        observed=np.atleast_2d(np.cov(x,bias=True))
        np.testing.assert_array_equal(w[k:,i],0)
        # Six standard errors for state means, 3% + absolute floor for cov.
        error=np.abs(x.mean(1)-counts)
        assert np.all(error <= 6*np.sqrt(n*p*(1-p)/b)+1e-10)
        np.testing.assert_allclose(observed,expected,rtol=.03,atol=.035)
        if k>1: assert np.all(observed[np.triu_indices(k,1)] < 0)
        exact_mu=(e*inv*x).sum(0)/n
        exact_var=((e**2-(1-q)*e)*inv2*x).sum(0)/n-exact_mu**2
        np.testing.assert_allclose(mu[i],exact_mu,rtol=2e-6,atol=2e-6)
        np.testing.assert_allclose(var[i],exact_var,rtol=2e-5,atol=2e-5)
        cpu_w=rng.multinomial(n,p,size=b).T
        cpu_mu=(e*inv*cpu_w).sum(0)/n
        cpu_var=((e**2-(1-q)*e)*inv2*cpu_w).sum(0)/n-cpu_mu**2
        ratios=[float(mu[i].std()/cpu_mu.std()),float(var[i].std()/cpu_var.std())] if k>1 else None
        if ratios: np.testing.assert_allclose(ratios,1,rtol=.02)
        report['sampler'].append({'k':k,'max_cov_error':float(np.max(abs(observed-expected))),
                                  'sd_ratios':ratios})
    # Nontrivial covariates, multiple treatments, missing groups and invalid
    # common draws. Compare all six summaries with the real sklearn reference.
    g,h,b=6,16,2000
    cov=rng.normal(size=(h,3)); cov=np.column_stack([np.ones(h),cov])
    treatment=rng.normal(size=(h,2)); weights=rng.integers(50,1000,h)
    mean=rng.normal(size=(g,h,b+1)); var=rng.normal(size=mean.shape)
    mean[1,2,4]=np.nan
    good=np.ones((g,h),bool); good[2:4,:3]=False
    tasks=[{'covariate':cov,'treatment':treatment,'Nc_list':weights} for _ in range(g)]
    ref=np.array([ht._regress_1d(cov[m],treatment[m],mean[i,m],var[i,m],weights[m])
                  for i,m in enumerate(good)])
    args=[torch.as_tensor(x,device='cuda') for x in (mean,var,good)]
    result=np.array(regress(*args,tasks))
    np.testing.assert_allclose(result,ref,rtol=1e-8,atol=1e-10,equal_nan=True)
    report['regression_max_abs_error']=float(np.nanmax(abs(result-ref)))
    if args_cli.prepared:
        with open(args_cli.prepared,'rb') as f: _,a,_,_=pickle.load(f)
        u=a.uns['memento']
        groups=sorted(u['groups'],key=lambda g:u['group_cells'][g].shape[0])
        report['real_states']=[]
        for group in (groups[0],groups[-1]):
            order=np.argsort(u['1d_moments'][group][0])
            for index in order[np.linspace(0,len(order)-1,4).astype(int)]:
                data=u['group_cells'][group][:,index]
                state=bootstrap._unique_expr(data,u['approx_size_factor'][group])
                mu,var,w=sample_states([state],data.shape[0],u['group_q'][group],40000,True)
                mu,var,w=mu[0].cpu().numpy(),var[0].cpu().numpy(),w[:,0].cpu().numpy()
                inv,inv2,e,counts=state
                exact_mu,exact_var=estimator._hyper_1d_relative(
                    data=(e,w.astype(np.float64)),n_obs=data.shape[0],q=u['group_q'][group],size_factor=(inv,inv2))
                cpu_mu,cpu_var=bootstrap._bootstrap_1d(data,u['approx_size_factor'][group],
                    u['group_q'][group],estimator._hyper_1d_relative,num_boot=40000,rng=rng)
                ratios=[float(mu.std()/cpu_mu.std()),float(var.std()/cpu_var.std())]
                precision=[float(np.max(abs(mu-exact_mu))/exact_mu.std()),
                           float(np.max(abs(var-exact_var))/exact_var.std())]
                np.testing.assert_array_equal(w.sum(0),data.shape[0])
                np.testing.assert_allclose(ratios,1,rtol=.035)
                assert max(precision)<.001, precision
                report['real_states'].append({'gene':a.var_names[index], 'group':group,
                    'k':len(counts),'sd_ratios':ratios,'max_roundoff_over_bootstrap_sd':precision,
                    'variance_sign_disagreements':int(((var>0)!=(exact_var>0)).sum())})
                print('validated',group,a.var_names[index],flush=True)
    if args_cli.out:
        with open(args_cli.out,'w') as f: json.dump(report,f,indent=2)
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
