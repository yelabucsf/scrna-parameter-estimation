import pandas as pd
import seaborn as sns
import scanpy as sc
import scipy.sparse as sparse
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import scipy.optimize as opt
import itertools

import scvi


def compare_estimators(smfish, fish_gapdh_sf, dropseq, dropseq_sf, overlap_genes,p=1):

    q=0.02*p
    mfish = (smfish/fish_gapdh_sf.reshape(-1,1))[overlap_genes].mean(axis=0)

    # Sample mean
    m1 = (dropseq/dropseq_sf.reshape(-1,1))[overlap_genes].mean(axis=0)
    
    # Sample sum divided by total
    m4= dropseq.sum(axis=0)/dropseq_sf.sum()

    # Filtering cells
    cells_with_many_genes = (dropseq_sf > 1500*p)
    m2 = (dropseq/dropseq_sf.reshape(-1,1)).loc[cells_with_many_genes,overlap_genes].mean(axis=0)
    m2 = np.maximum(m1,m2)

    # Solve for optimal weights
    X = dropseq/dropseq_sf.reshape(-1,1)
    naive_v = X.var(axis=0)[overlap_genes]
    v = naive_v-(1-q)*(dropseq[overlap_genes].values/(dropseq_sf**2-dropseq_sf*(1-q)).reshape(-1,1)).mean(axis=0)
    variance_contributions = ((1-q)/dropseq_sf).reshape(-1,1)*m1.values.reshape(1,-1) + v.values.reshape(1,-1)
    m3 = pd.Series(np.average( (dropseq/dropseq_sf.reshape(-1,1))[overlap_genes], weights=1/variance_contributions, axis=0), index=m2.index)
    m3[m3<0] = m2[m3<0]
    
    return mfish, m1, m2, m3



data_path = '/data_volume/memento/saver/'

dropseq = pd.read_csv('/data_volume/memento/saver/melanoma_dropseq.csv', index_col=0, sep=',').T
# dropseq = dropseq[dropseq['GAPDH'] > 0]
dropseq_sf = dropseq.sum(axis=1).values


smfish = pd.read_csv('/data_volume/memento/saver/fishSubset.txt', index_col=0, sep=' ')
# smfish = smfish[smfish['GAPDH'] > 0]
fish_gapdh_sf = (smfish['GAPDH']+1).values
# smfish_normalized = smfish#/fish_gapdh_sf.reshape(-1, 1)
# smfish = pd.read_csv('fishSubset (1).txt', sep=' ', index_col=0)

overlap_genes = list(set(dropseq.columns) & set(smfish.columns))
overlap_genes = dropseq.mean(axis=0)[overlap_genes][dropseq.mean(axis=0)[overlap_genes] > 0.02].index.tolist()

smfish = smfish[overlap_genes].fillna(0.0)

scvi_result = []
mfish = (smfish/fish_gapdh_sf.reshape(-1,1))[overlap_genes].mean(axis=0)
for num_cell in [50, 100, 200, 300, 500, 1000]:
        
        for trial in range(100):
            
            print(num_cell, trial)
            sample_idx = np.random.choice(dropseq.shape[0], num_cell)
            tiny = dropseq.iloc[sample_idx]
            shallow = tiny
            shallow_sf = shallow.sum(axis=1).values
            mean_numi = shallow_sf.mean()
            
            adata = sc.AnnData(X=shallow.values, obs=pd.DataFrame(index=shallow.index), var=pd.DataFrame(index=shallow.columns))
            scvi.model.SCVI.setup_anndata(
                adata,
                categorical_covariate_keys=[],
                continuous_covariate_keys=[],
            )
            model = scvi.model.SCVI(adata)
            model.train()
            scvi_denoised = model.get_normalized_expression(library_size=10e4).mean(axis=0)
            
            corr1 = stats.pearsonr(np.log(mfish), np.log(scvi_denoised[overlap_genes]))[0]
            scvi_result.append((num_cell, mean_numi, corr1))


scvi_result = pd.DataFrame(scvi_result, columns=['num_cell', 'mean_umi', 'scvi'])
scvi_result.to_csv(data_path + 'numcells_scvi.csv', index=False)