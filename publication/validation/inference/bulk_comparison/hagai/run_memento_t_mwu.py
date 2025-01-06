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
sys.path.append('/home/ubuntu/Github/scrna-parameter-estimation/')

import memento

logging.basicConfig(
    format="%(asctime)s %(process)-7s %(levelname)-8s %(message)s",
    level=logging.INFO, 
    datefmt="%Y-%m-%d %H:%M:%S",
)
logging.captureWarnings(True)

data_path = '/data_volume/bulkrna/hagai/sc_rnaseq/'

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

def run_t_mwu():
    
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

        
def run_memento():
    
    for file in files:
        
        condition = file.split('-')[-1] + '4'
        
        adata = sc.read_h5ad(data_path +'h5Seurat/' + file + '.h5ad')
        
        gene_list = adata.var.index[adata.X.mean(axis=0).A1 > 0.02].tolist()

        adata.obs['q'] = 0.07
        memento.setup_memento(
            adata, 
            'q', 
            filter_mean_thresh=0.00, 
            estimator_type='pseudobulk')
        
        memento.create_groups(adata, label_columns=['replicate', 'label'])

        memento.compute_1d_moments(adata, min_perc_group=0, gene_list=gene_list)

        condition = 'unst'
        df = pd.DataFrame(index=adata.uns['memento']['groups'])
        df['mouse'] = df.index.str.split('^').str[1]
        df['stim'] = df.index.str.split('^').str[2]
        cov_df = pd.get_dummies(df[['mouse']], drop_first=True).astype(float)
        stim_df = (df[['stim']]==condition).astype(float)
        cov_df = sm.add_constant(cov_df)
        cov_df = cov_df[['const']]
        
        memento.ht_mean(
            adata=adata, 
            treatment=stim_df,
            covariate=cov_df,
            treatment_for_gene=None,
            covariate_for_gene=None,
            inplace=True, 
            num_boot=2000, 
            verbose=1,
            num_cpus=14)
        
        results = memento.get_mean_ht_result(adata)
        results.columns = ['gene', 'tx', 'coef', 'se', 'pval'] # for standardization purposes
        results.set_index('gene', inplace=True)
        results['fdr'] = memento.util._fdrcorrect(results['pval'])
        results.to_csv(data_path + f'results/{file}_quasiML.csv')            
        
if __name__ == '__main__':
    
    # logging.info('t, mwu')
    # run_t_mwu()
    
    logging.info('memento')
    run_memento()
    