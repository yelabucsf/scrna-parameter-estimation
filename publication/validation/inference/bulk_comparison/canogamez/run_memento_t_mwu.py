import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import functools
import numpy as np
import scanpy as sc
import scipy.stats as stats
from statsmodels.stats.multitest import fdrcorrection
from patsy import dmatrix, dmatrices 
import statsmodels.api as sm
import logging
import pickle as pkl
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

data_path = '/data_volume/bulkrna/canogamez/'

datasets = [
 'CD4_Memory-Th0',
 'CD4_Memory-Th2',
 'CD4_Memory-Th17',
 'CD4_Memory-iTreg',
 'CD4_Naive-Th0',
 'CD4_Naive-Th2',
 'CD4_Naive-Th17',
 'CD4_Naive-iTreg']

columns = ['coef', 'pval', 'fdr']

def safe_fdr(x):
    fdr = np.ones(x.shape[0])
    _, fdr[np.isfinite(x)] = fdrcorrection(x[np.isfinite(x)])
    return fdr


def run_memento():
    
    for trial in [1]:
        
        for dataset in datasets: 

            logging.info(f'Processing {dataset} dataset')
            adata = sc.read_h5ad(data_path + 'single_cell/{}_{}.h5ad'.format(dataset, trial))

            gene_list = adata.var.index[adata.X.mean(axis=0).A1 > 0.02].tolist()

            adata.obs['q'] = 0.07
            memento.setup_memento(
                adata, 
                'q', 
                filter_mean_thresh=0.02, 
                estimator_type='pseudobulk')

            memento.create_groups(adata, label_columns=['donor.id', 'cytokine.condition'])

            memento.compute_1d_moments(adata, min_perc_group=0, gene_list=gene_list)

            condition = 'UNS'
            df = pd.DataFrame(index=adata.uns['memento']['groups'])
            df['mouse'] = df.index.str.split('^').str[1]
            df['stim'] = df.index.str.split('^').str[2]
            cov_df = pd.get_dummies(df[['mouse']], drop_first=True).astype(float)
            stim_df = (df[['stim']]==condition).astype(float)
            cov_df = sm.add_constant(cov_df)

            memento.ht_mean(
                adata=adata, 
                treatment=stim_df,
                covariate=cov_df[['const']],
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
            results.to_csv(data_path + 'sc_results/{}_{}_quasiML.csv'.format(dataset, trial))
            
def run_t_mwu():
    
    for trial in [1, 100]:
        
        for dataset in datasets: 

            print(trial, dataset)

            ct, stim = dataset.split('-')

            adata = sc.read(data_path + 'single_cell/{}_{}.h5ad'.format(dataset, trial))

            labels = adata.obs['cytokine.condition'].drop_duplicates().tolist()

            data1 = adata[adata.obs['cytokine.condition'] ==labels[0]].X.todense()
            data2 = adata[adata.obs['cytokine.condition'] ==labels[1]].X.todense()

            statistic, pvalue = stats.ttest_ind(data1, data2, axis=0)

            logfc = data1.mean(axis=0) - data2.mean(axis=0)

            ttest_result = pd.DataFrame(
                zip(logfc.A1, pvalue, safe_fdr(pvalue)), 
                index=adata.var.index,
                columns=columns)
            ttest_result.to_csv(data_path + f'sc_results/{dataset}_{trial}_t.csv')

            mwu_stat, mwu_pval = stats.mannwhitneyu(data1, data2, axis=0)
            mwu_result = pd.DataFrame(
                zip(logfc.A1, mwu_pval, safe_fdr(mwu_pval)), 
                index=adata.var.index,
                columns=columns)
            mwu_result.to_csv(data_path + f'sc_results/{dataset}_{trial}_MWU.csv')

    
if __name__ == '__main__':
    
    logging.info('Running single cell methods (memento, t-test, MWU) for Cano-Gamez datasets')
    run_memento()
    
    # run_t_mwu()
        