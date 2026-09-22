"""Optional backend tests. CUDA tests skip cleanly without the optional extra."""
import subprocess
import sys
import numpy as np
import pandas as pd
import pytest
from scipy import sparse
import anndata as ad
import memento


def test_cpu_import_does_not_import_torch():
    subprocess.run([sys.executable,'-c',
                    "import sys; import memento; assert 'torch' not in sys.modules"],check=True)


def test_invalid_backend():
    with pytest.raises(ValueError,match='backend'):
        memento.ht_1d_moments(None,None,backend='invalid')


@pytest.fixture
def cuda(request):
    torch=pytest.importorskip('torch')
    if not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    if request.config._memento_torch_threads is None:
        request.config._memento_torch_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    from memento import _gpu
    return torch,_gpu


@pytest.fixture
def prepared():
    from memento.estimator import _hyper_1d_relative
    rng=np.random.default_rng(17)
    groups=[f'd{d}_{c}' for d in range(3) for c in ('ctrl','stim')]
    cells={};sf={};moments={}
    for i,g in enumerate(groups):
        x=rng.negative_binomial(2,.4,size=(48,5))
        if i%2:x[:,0]+=rng.poisson(2,size=48)
        cells[g]=sparse.csc_matrix(x)
        sf[g]=rng.choice([1.,1.2,1.5],48)
        m,v=_hyper_1d_relative(cells[g],48,.07,sf[g])
        moments[g]=[m,v,v]
    a=ad.AnnData(sparse.vstack(list(cells.values())).tocsr(),var=pd.DataFrame(index=[f'g{i}' for i in range(5)]))
    a.uns['memento']={'groups':groups,'group_cells':cells,'approx_size_factor':sf,
        '1d_moments':moments,'group_q':dict.fromkeys(groups,.07),
        'mv_regressor':{g:np.zeros(3) for g in groups},'estimator_type':'hyper_relative'}
    tx=pd.DataFrame({'stim':[0,1]*3,'interaction':[0,-1,0,0,0,1]},index=groups,dtype=float)
    cov=pd.DataFrame({'d1':[0,0,1,1,0,0],'d2':[0,0,0,0,1,1]},index=groups,dtype=float)
    return a,tx,cov


def test_weights_and_fixed_weight_moments(cuda,prepared):
    torch,gpu=cuda;a,_,_=prepared
    gen=gpu._generator('cuda',101)
    w=gpu._weights(7,50000,'cuda',gen).cpu().numpy()
    np.testing.assert_array_equal(w.sum(1),7)
    np.testing.assert_allclose(w.mean(0),1,atol=.025,rtol=0)
    np.testing.assert_allclose(np.cov(w.T,bias=True),np.eye(7)-np.ones((7,7))/7,atol=.025,rtol=0)
    u=a.uns['memento'];g=u['groups'][0];x=u['group_cells'][g].toarray();sf=u['approx_size_factor'][g]
    w=gpu._weights(len(x),512,'cuda',gen)
    coef=gpu._coefficients(x,sf,.07,'cuda')
    got=(w@coef).cpu().numpy()
    inv=1/sf
    ref=w.cpu().numpy().astype(float)@np.concatenate((x*inv[:,None],(x*x-.93*x)*(inv**2)[:,None]),axis=1)
    np.testing.assert_allclose(got,ref,rtol=2e-6,atol=1e-5)


def test_regress_matches_cpu_with_masks_designs_and_nonfinite_draws(cuda,prepared):
    torch,gpu=cuda;_,tx,cov=prepared
    from memento.hypothesis_test import _regress_1d
    rng=np.random.default_rng(15)
    m=rng.normal(size=(3,6,101));v=rng.normal(size=m.shape)
    m[1,2,8]=np.inf
    good=np.ones((3,6),bool);good[2,1]=False
    designs=[(('stim',),('d1','d2')),(('interaction','stim'),('d1','d2')),(('stim',),('d1',))]
    nc=np.arange(6)+20
    got=gpu._regress(torch.tensor(m,device='cuda'),torch.tensor(v,device='cuda'),
                     torch.tensor(good,device='cuda'),cov,tx,nc,designs)
    for i,(ts,cs) in enumerate(designs):
        mask=good[i]
        ref=_regress_1d(cov[list(cs)].values[mask],tx[list(ts)].values[mask],m[i,mask],v[i,mask],nc[mask])
        np.testing.assert_allclose(got[i],ref,rtol=1e-10,atol=1e-12,equal_nan=True)


def test_api_seed_order_inplace_and_no_compression(cuda,prepared,monkeypatch):
    torch,gpu=cuda;a,tx,cov=prepared
    from memento import bootstrap,main
    def forbidden(*args,**kwargs):raise AssertionError('GPU called compressed sampling or CPU job dispatcher')
    monkeypatch.setattr(bootstrap,'_unique_expr',forbidden)
    monkeypatch.setattr(main,'Parallel',forbidden)
    assignment={'g4':['interaction','stim'],'g1':['stim']}
    covs={'g4':['d1','d2'],'g1':['d1']}
    torch.cuda.init();before=torch.cuda.get_rng_state().clone()
    original_tf32=torch.backends.cuda.matmul.allow_tf32
    kwargs=dict(treatment=tx,covariate=cov,treatment_for_gene=assignment,covariate_for_gene=covs,
                num_boot=1000,backend='gpu',inplace=False,random_state=71,gpu_batch_size=1,verbose=0)
    first=memento.ht_1d_moments(a,**kwargs);second=memento.ht_1d_moments(a,**kwargs)
    assert '1d_ht' not in a.uns['memento']
    r=memento.get_1d_ht_result(first)
    assert list(zip(r.gene,r.tx))==[('g4','interaction'),('g4','stim'),('g1','stim')]
    np.testing.assert_array_equal(r.iloc[:,2:],memento.get_1d_ht_result(second).iloc[:,2:])
    assert torch.equal(before,torch.cuda.get_rng_state())
    assert torch.backends.cuda.matmul.allow_tf32==original_tf32
    assert np.isfinite(r.iloc[:,2:].values).all()


def test_streaming_policy_and_no_group_identifiability(cuda,prepared):
    torch,gpu=cuda;a,tx,cov=prepared
    got=memento.ht_1d_moments(a,treatment=tx[['stim']],covariate=cov,num_boot=1024,
        backend='gpu',gpu_memory_budget=1,gpu_batch_size=1,inplace=False,verbose=0)
    assert not got.uns['memento']['1d_ht']['gpu']['cached_cell_weights']
    assert np.isfinite(memento.get_1d_ht_result(got).iloc[:,2:].values).all()
    good=torch.zeros((1,6),device='cuda',dtype=torch.bool)
    y=torch.ones((1,6,4),device='cuda',dtype=torch.float64)
    out=gpu._regress(y,y,good,cov,tx,np.ones(6),[(('stim',),('d1','d2'))])
    assert np.isnan(out).all()


def test_empty_tests(cuda,prepared):
    _,_=cuda;a,tx,cov=prepared
    got=memento.ht_1d_moments(a,treatment=tx,covariate=cov,treatment_for_gene={},backend='gpu',inplace=False)
    assert memento.get_1d_ht_result(got).empty


@pytest.mark.parametrize('options,error',[
    ({'resample_rep':True},NotImplementedError),({'approx':'gdp'},NotImplementedError),
    ({'gpu_memory_budget':0},ValueError),({'gpu_batch_size':0},ValueError),
    ({'num_boot':1},ValueError),({'gpu_device':'cpu'},ValueError),({'unknown_option':1},TypeError)])
def test_unsupported_options_are_explicit(cuda,prepared,options,error):
    a,tx,cov=prepared
    with pytest.raises(error):
        memento.ht_1d_moments(a,treatment=tx,covariate=cov,backend='gpu',**options)
    assert '1d_ht' not in a.uns['memento']


def test_invalid_counts_and_estimator(cuda,prepared):
    a,tx,cov=prepared
    a.uns['memento']['estimator_type']='poisson_relative'
    with pytest.raises(NotImplementedError):memento.ht_1d_moments(a,tx,cov,backend='gpu')
    a.uns['memento']['estimator_type']='hyper_relative'
    group=a.uns['memento']['groups'][0]
    a.uns['memento']['group_cells'][group]=a.uns['memento']['group_cells'][group].astype(float)
    a.uns['memento']['group_cells'][group].data[0]=.5
    with pytest.raises(ValueError,match='integer expression'):
        memento.ht_1d_moments(a,tx,cov,backend='gpu')


@pytest.fixture
def prepared_pairs(prepared):
    a,tx,cov=prepared
    u=a.uns['memento']
    u['size_factor']=u['approx_size_factor']
    memento.compute_2d_moments(a,[('g0','g1'),('g1','g2'),('g3','g4'),('g0','g0')])
    return a,tx,cov


def test_pair_fixed_weight_correlations(cuda,prepared_pairs):
    torch,gpu=cuda;a,_,_=prepared_pairs
    from memento.estimator import _hyper_1d_relative,_hyper_cov_relative,_corr_from_cov
    u=a.uns['memento'];group=u['groups'][0]
    x=u['group_cells'][group][:,[0,1]].toarray();sf=u['approx_size_factor'][group]
    w=gpu._weights(len(x),1000,'cuda',gpu._generator('cuda',18))
    got=gpu._pair_correlations(w@gpu._pair_coefficients(x,sf,.07,'cuda'),len(x),1).cpu().numpy()[0]
    weights=w.cpu().numpy().T.astype(float)
    inv=(1/sf)[:,None];inv2=inv**2
    one=x[:,0,None];two=x[:,1,None]
    _,v1=_hyper_1d_relative((one,weights),len(x),.07,(inv,inv2))
    _,v2=_hyper_1d_relative((two,weights),len(x),.07,(inv,inv2))
    c=_hyper_cov_relative((one,two,weights),len(x),(inv,inv2),.07)
    ref=_corr_from_cov(c,v1,v2,boot=True)
    np.testing.assert_allclose(got,ref,rtol=2e-4,atol=3e-6,equal_nan=True)
    vals=torch.tensor([[-1.,1.,float('nan'),.2,.8],[2.,-2.,float('nan'),1.,-1.]],device='cuda')
    filled,valid=gpu._fill_valid(vals,torch.isfinite(vals)&(vals.abs()<1),gpu._generator('cuda',1))
    assert valid.tolist()==[True,False]
    assert set(filled[0].tolist()).issubset(set(vals[0,3:].tolist()))


def test_correlation_regression_matches_cpu(cuda,prepared):
    torch,gpu=cuda;_,tx,cov=prepared
    from memento.hypothesis_test import _regress_2d
    rng=np.random.default_rng(191)
    y=rng.uniform(-.9,.9,(3,6,1001));y[0,2,99]=np.nan
    good=np.ones((3,6),bool);good[1,1]=False;good[2]=False
    designs=[(('stim','interaction'),('d1','d2')),(('stim',),('d1',)),(('stim',),('d1','d2'))]
    nc=np.arange(6)+30
    got=gpu._regress(torch.tensor(y,device='cuda'),None,torch.tensor(good,device='cuda'),cov,tx,nc,designs)
    for i in (0,1):
        ts,cs=designs[i];mask=good[i]
        ref=_regress_2d(cov[list(cs)].values[mask],tx[list(ts)].values[mask],y[i,mask],nc[mask])
        np.testing.assert_allclose(got[i],ref,atol=1e-12,rtol=1e-10)
    assert np.isnan(got[2]).all()


def test_pair_api_order_reproducibility_streaming_and_diagonal(cuda,prepared_pairs,monkeypatch):
    torch,gpu=cuda;a,tx,cov=prepared_pairs
    from memento import bootstrap,main
    def forbidden(*args,**kwargs):raise AssertionError('Compressed CPU path called')
    monkeypatch.setattr(bootstrap,'_bootstrap_2d',forbidden)
    monkeypatch.setattr(main,'Parallel',forbidden)
    kwargs=dict(treatment=tx,covariate=cov,backend='gpu',inplace=False,num_boot=1024,
                gpu_memory_budget=1,gpu_batch_size=1,random_state=9)
    before=torch.cuda.get_rng_state().clone()
    first=memento.ht_2d_moments(a,**kwargs);second=memento.ht_2d_moments(a,**kwargs)
    assert '2d_ht' not in a.uns['memento']
    assert not first.uns['memento']['2d_ht']['gpu']['cached_cell_weights']
    r=memento.get_2d_ht_result(first)
    np.testing.assert_array_equal(r.iloc[:,3:],memento.get_2d_ht_result(second).iloc[:,3:])
    assert np.isfinite(r.iloc[:6,3:].values).all()
    assert np.isnan(r.iloc[6:,3:].values).all()
    assert torch.equal(before,torch.cuda.get_rng_state())
    assignments={('g3','g4'):['interaction','stim'],('g0','g1'):['stim']}
    covs={('g3','g4'):['d1'],('g0','g1'):['d1','d2']}
    result=memento.ht_2d_moments(a,treatment_for_gene=assignments,covariate_for_gene=covs,**kwargs)
    r=memento.get_2d_ht_result(result)
    assert list(zip(r.gene_1,r.gene_2,r.tx))==[('g0','g1','stim'),('g3','g4','interaction'),('g3','g4','stim')]
    empty=memento.ht_2d_moments(a,treatment_for_gene={},**kwargs)
    assert memento.get_2d_ht_result(empty).empty
    with pytest.raises(NotImplementedError):memento.ht_2d_moments(a,resample_rep=True,**kwargs)


def test_automatic_memory_policy(cuda):
    _,gpu=cuda
    gib=1024**3
    for available in (256*1024**2,2*gib,8*gib,24*gib,80*gib):
        budget=gpu._working_budget(available,None)
        assert budget<=available*.5 and budget<=8*gib
        batch,chunk,cache,retained=gpu._memory_plan(np.array([100,200,300,400]),10000,1000,budget,256)
        assert 1<=batch<=256 and 1<=chunk<=512
        assert retained<=budget//3
    assert gpu._working_budget(gib,2*gib)==int(.75*gib)


def test_wide_eqtl_regression_matches_cpu(cuda,prepared,monkeypatch):
    torch,gpu=cuda;_,_,cov=prepared
    from memento.hypothesis_test import _regress_1d
    rng=np.random.default_rng(21)
    tx=pd.DataFrame(rng.integers(0,3,(6,131)),columns=[f'rs{i}' for i in range(131)],dtype=float)
    # Remove monomorphic columns exactly as the public API does.
    tx=tx.loc[:,tx.nunique()>1]
    designs=[(tuple(tx.columns),tuple(cov.columns)),(tuple(tx.columns[::3]),('d1',))]
    m=rng.normal(size=(2,6,501));v=rng.normal(size=m.shape)
    good=np.ones((2,6),bool);nc=np.arange(6)+30
    original=gpu._summarize_block;seen=[]
    def checked(mapping,ys,observed=False):
        seen.append(mapping.shape[0]);assert mapping.shape[0]<=6
        return original(mapping,ys,observed)
    monkeypatch.setattr(gpu,'_summarize_block',checked)
    got=gpu._regress(torch.tensor(m,device='cuda'),torch.tensor(v,device='cuda'),
        torch.tensor(good,device='cuda'),cov,tx,nc,designs)
    for i,(ts,cs) in enumerate(designs):
        ref=_regress_1d(cov[list(cs)].values,tx[list(ts)].values,m[i],v[i],nc)
        identifiable=np.isfinite(gpu._regression_map(cov[list(cs)].values,tx[list(ts)].values,nc)).all(1)
        np.testing.assert_allclose(np.asarray(got[i])[:,identifiable],np.asarray(ref)[:,identifiable],
                                   rtol=1e-9,atol=1e-11,equal_nan=True)
        assert np.isnan(got[i][:,~identifiable]).all()
    assert len(seen)>20
    # Donor genotype is not identifiable after conditioning on donor indicators.
    confounded=np.array([[0],[0],[1],[1],[2],[2]],dtype=float)
    assert np.isnan(gpu._regression_map(cov.values,confounded,nc)).all()


def test_eqtl_dictionary_monomorphic_and_auto_budget(cuda,prepared):
    _,gpu=cuda;a,_,_=prepared
    tx=pd.DataFrame({'rsA':[0,0,1,1,2,2],'rsB':[2,2,0,0,1,1],'mono':[0]*6},dtype=float)
    cov=pd.DataFrame({'condition':[0,1]*3},dtype=float)
    assignment={'g3':['rsB','mono','rsA'],'g1':['rsA'],'g0':['mono']}
    a=memento.ht_1d_moments(a,tx,cov,treatment_for_gene=assignment,backend='gpu',num_boot=1000,inplace=False)
    r=memento.get_1d_ht_result(a)
    assert list(zip(r.gene,r.tx))==[('g3','rsB'),('g3','rsA'),('g1','rsA')]
    assert np.isfinite(r.iloc[:,2:].values).all()
    assert a.uns['memento']['1d_ht']['gpu']['memory_policy']=='auto'


def test_missing_torch_is_optional_and_gpu_error_is_actionable():
    code = '''
import importlib.abc
import sys
class NoTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'torch' or fullname.startswith('torch.'):
            raise ModuleNotFoundError("No module named 'torch'", name='torch')
sys.meta_path.insert(0, NoTorch())
import memento
assert 'torch' not in sys.modules
try:
    import memento._gpu
except ImportError as exc:
    assert 'memento-de[gpu]' in str(exc)
else:
    raise AssertionError('GPU import unexpectedly succeeded without torch')
'''
    subprocess.run([sys.executable, '-c', code], check=True)


@pytest.mark.parametrize('dimension',[1,2])
def test_unavailable_cuda_is_explicit(cuda,prepared_pairs,monkeypatch,dimension):
    torch,_=cuda;a,tx,cov=prepared_pairs
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    function=memento.ht_1d_moments if dimension==1 else memento.ht_2d_moments
    with pytest.raises(RuntimeError,match='available CUDA device'):
        function(a,tx,cov,backend='gpu')
    assert f'{dimension}d_ht' not in a.uns['memento']


@pytest.mark.parametrize('dimension',[1,2])
def test_failure_restores_cuda_state_and_does_not_publish_results(cuda,prepared_pairs,monkeypatch,dimension):
    torch,gpu=cuda;a,tx,cov=prepared_pairs
    before=torch.cuda.get_rng_state().clone()
    previous=torch.backends.cuda.matmul.allow_tf32
    def failed(*args,**kwargs):raise RuntimeError('injected regression failure')
    monkeypatch.setattr(gpu,'_regress',failed)
    function=memento.ht_1d_moments if dimension==1 else memento.ht_2d_moments
    try:
        torch.backends.cuda.matmul.allow_tf32=True
        with pytest.raises(RuntimeError,match='injected regression failure'):
            function(a,tx,cov,backend='gpu',num_boot=32)
        assert torch.backends.cuda.matmul.allow_tf32
        assert torch.equal(before,torch.cuda.get_rng_state())
        assert f'{dimension}d_ht' not in a.uns['memento']
    finally:
        torch.backends.cuda.matmul.allow_tf32=previous


def test_streamed_weights_are_shared_across_gene_batches(cuda,prepared,monkeypatch):
    torch,gpu=cuda;a,tx,cov=prepared
    original=gpu._weights;draws=[]
    def capture(*args,**kwargs):
        w=original(*args,**kwargs);draws.append(w.cpu().numpy());return w
    monkeypatch.setattr(gpu,'_weights',capture)
    result=memento.ht_1d_moments(a,tx[['stim']],cov,backend='gpu',num_boot=1024,
        gpu_memory_budget=1,gpu_batch_size=1,inplace=False,treatment_for_gene={'g0':['stim'],'g1':['stim']})
    assert not result.uns['memento']['1d_ht']['gpu']['cached_cell_weights']
    assert len(draws)%2==0
    half=len(draws)//2
    for first,second in zip(draws[:half],draws[half:]):
        np.testing.assert_array_equal(first,second)


def test_all_invalid_bootstrap_groups_return_nan(cuda,prepared_pairs):
    _,_=cuda;a,tx,cov=prepared_pairs
    # True moments can be valid while a degenerate resampling population has
    # no valid corrected variances/correlations. Neither path may invent draws.
    for group in a.uns['memento']['groups']:
        a.uns['memento']['group_cells'][group]=sparse.csc_matrix((48,5),dtype=float)
    for dimension,function in [(1,memento.ht_1d_moments),(2,memento.ht_2d_moments)]:
        result=function(a,tx,cov,backend='gpu',num_boot=32,inplace=False)
        getter=memento.get_1d_ht_result if dimension==1 else memento.get_2d_ht_result
        frame=getter(result)
        assert np.isnan(frame.iloc[:,2 if dimension==1 else 3:].values).all()
