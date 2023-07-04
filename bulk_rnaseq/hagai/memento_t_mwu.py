import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import functools
import numpy as np
import scanpy as sc
import logging
import scipy.stats as stats
from statsmodels.stats.multitest import fdrcorrection
from patsy import dmatrix, dmatrices 
import statsmodels.api as sm

import sys
sys.path.append('/home/ssm-user/Github/memento')

import memento.model.rna as rna
import memento.estimator.hypergeometric as hg

logging.basicConfig(
    format="%(asctime)s %(process)-7s %(levelname)-8s %(message)s",
    level=logging.INFO, 
    datefmt="%Y-%m-%d %H:%M:%S",
)
logging.captureWarnings(True)

data_path = '/data_volume/memento/hagai/sc_rnaseq/'

columns = ['coef', 'pval', 'fdr']

files = [
    'Hagai2018_mouse-lps',
    'Hagai2018_mouse-pic',
    'Hagai2018_pig-lps',
    'Hagai2018_rabbit-lps',
    'Hagai2018_rat-lps',
    'Hagai2018_rat-pic',
]

def safe_fdr(x):
    fdr = np.ones(x.shape[0])
    _, fdr[np.isfinite(x)] = fdrcorrection(x[np.isfinite(x)])
    return fdr

def run_t_mwu(shuffled=False):
    
    out_suffix = '_shuffled' if shuffled else ''

    for file in files:

        print('working on', file)

        adata = sc.read_h5ad(data_path +'h5Seurat/' + file + '.h5ad')

        if shuffled:
            adata.obs['label'] = adata.obs['label'].sample(frac=1).values

        labels = adata.obs['label'].drop_duplicates().tolist()

        data1 = adata[adata.obs['label'] ==labels[0]].X.todense()
        data2 = adata[adata.obs['label'] ==labels[1]].X.todense()

        statistic, pvalue = stats.ttest_ind(data1, data2, axis=0)

        logfc = data1.mean(axis=0) - data2.mean(axis=0)

        ttest_result = pd.DataFrame(
            zip(logfc.A1, pvalue, safe_fdr(pvalue)), 
            index=adata.var.index,
            columns=columns)
        ttest_result.to_csv(data_path + f'results/{file}{out_suffix}_t.csv')

        mwu_stat, mwu_pval = stats.mannwhitneyu(data1, data2, axis=0)
        mwu_result = pd.DataFrame(
            zip(logfc.A1, mwu_pval, safe_fdr(mwu_pval)), 
            index=adata.var.index,
            columns=columns)
        mwu_result.to_csv(data_path + f'results/{file}{out_suffix}_MWU.csv')

        
def run_memento(shuffled=False):
    
    out_suffix = '_shuffled' if shuffled else ''
    
    for file in files:
        
        adata = sc.read_h5ad(data_path +'h5Seurat/' + file + '.h5ad')
        adata.obs['q'] = 0.07

        if shuffled:
            adata.obs['label'] = adata.obs['label'].sample(frac=1).values

        rna.MementoRNA.setup_anndata(
            adata=adata,
            q_column='q',
            label_columns=['replicate', 'label'],
            num_bins=30)
        
        adata.var['expr_genes'] = (adata.X.mean(axis=0).A1 > 0.02)
        adata = adata[:, adata.var['expr_genes']]
        
        model = rna.MementoRNA(adata=adata)

        model.compute_estimate(
            estimand='mean',
            get_se=True,
            n_jobs=30,
        )
        
        df = pd.DataFrame(index=adata.uns['memento']['groups'])
        df['mouse'] = df.index.str.split('^').str[1]
        df['stim'] = df.index.str.split('^').str[2]

        cov_df = pd.get_dummies(df[['mouse']], drop_first=True).astype(float)
        stim_df = (df[['stim']]=='lps4').astype(float)
        interactions = cov_df * stim_df[['stim']].values
        interactions.columns = ['stim*'+col for col in interactions.columns]
        cov_df = pd.concat([cov_df, interactions], axis=1)
        cov_df = sm.add_constant(cov_df)
        
        result = model.differential_mean(
            covariates=cov_df, 
            treatments=stim_df,
            estimator='log_mean',
            family='WLS',
            verbose=2,
            n_jobs=5)

        result['z'] = result['coef']/result['se']
        result['z_abs'] = result['z'].abs()
        result['pval'] = 2*stats.norm.sf(result['z'].abs())
        _, result['fdr'] = fdrcorrection(result['pval'])
        
        result.to_csv(data_path + f'results/{file}{out_suffix}_memento.csv')
        
        
if __name__ == '__main__':
    
    logging.info('t, mwu')
    run_t_mwu(shuffled=False)
    
    logging.info('t, mwu, shuffled')
    run_t_mwu(shuffled=True)
    
    logging.info('memento')
    run_memento(shuffled=False)
    
    logging.info('memento, shuffled')
    run_memento(shuffled=True)