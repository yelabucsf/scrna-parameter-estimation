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
import os

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

def run_t_mwu(num_cells=None):
    
    out_suffix = '_100' if num_cells is not None else ''

    for file in files:

        print('working on', file)

        adata = sc.read_h5ad(data_path +'h5Seurat/' + file + out_suffix + '.h5ad')

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

        
def run_memento(num_cells=None):
    
    out_suffix = '_100' if num_cells is not None else ''
    print('num cells', num_cells)
    
    for file in files:
        
        condition = file.split('-')[-1] + '4'
        
        dispersion = pd.read_csv(data_path + f'results/{file}_dispersions.csv', index_col=0)
        dispersion = dispersion.set_index('gene')
        
        adata = sc.read_h5ad(data_path +'h5Seurat/' + file + out_suffix + '.h5ad')
        adata.obs['q'] = 0.07

        rna.MementoRNA.setup_anndata(
            adata=adata,
            q_column='q',
            label_columns=['replicate', 'label'],
            num_bins=30)
        
        adata.var['expr_genes'] = (adata.X.mean(axis=0).A1 > 0.02)
        adata = adata[:, adata.var['expr_genes']]
        
        model = rna.MementoRNA(adata=adata)

        model_save_name = f'{file}_estimates'

        if model_save_name in os.listdir():
            model.load_estimates(model_save_name)
        else:
            model.compute_estimate(
                estimand='mean',
                get_se=True,
                n_jobs=30,
            )
            model.save_estimates(model_save_name)
        
        df = pd.DataFrame(index=adata.uns['memento']['groups'])
        df['mouse'] = df.index.str.split('^').str[1]
        df['stim'] = df.index.str.split('^').str[2]
        cov_df = pd.get_dummies(df[['mouse']], drop_first=True).astype(float)
        stim_df = (df[['stim']]==condition).astype(float)
        cov_df = sm.add_constant(cov_df)
        
        result = model.differential_mean(
            covariates=cov_df, 
            treatments=stim_df,
            family='quasiGLM',
            dispersions=dispersion.iloc[:, 0],
            verbose=2,
            n_jobs=10)
        
        _, result['fdr'] = fdrcorrection(result['pval'])
        result.to_csv(data_path + f'results/{file}{out_suffix}_quasiGLM.csv')                
        
if __name__ == '__main__':
    
    logging.info('t, mwu')
    run_t_mwu()
    
    logging.info('t, mwu, sampled')
    run_t_mwu(num_cells=100)
    
    logging.info('memento')
    run_memento()
    
    logging.info('memento, sampled')
    run_memento(num_cells=100)