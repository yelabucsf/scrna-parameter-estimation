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

from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats
from pydeseq2.utils import load_example_data

import sys
sys.path.append('/home/ssm-user/Github/memento')

import memento.model.rna as rna
import memento.estimator.hypergeometric as hg
import memento.util as util

logging.basicConfig(
    format="%(asctime)s %(process)-7s %(levelname)-8s %(message)s",
    level=logging.INFO, 
    datefmt="%Y-%m-%d %H:%M:%S",
)
logging.captureWarnings(True)

data_path = '/data_volume/memento/canogamez/'


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
    
    for trial in [1, 100]:
        
        for dataset in datasets: 

            print(trial, dataset)

            ct, stim = dataset.split('-')

            adata = sc.read(data_path + 'single_cell/{}_{}.h5ad'.format(dataset, trial))

            adata.obs['q'] = 0.07
            adata.X = adata.X.astype(float)

            rna.MementoRNA.setup_anndata(
                    adata=adata,
                    q_column='q',
                    label_columns=['donor.id', 'cytokine.condition'],
                    num_bins=30)

            adata = adata[:, adata.X.mean(axis=0).A1 > 0.02]

            model = rna.MementoRNA(adata=adata)

            model.compute_estimate(
                estimand='mean',
                get_se=True,
                n_jobs=30,
            )

            df = pd.DataFrame(index=adata.uns['memento']['groups'])
            df['mouse'] = df.index.str.split('^').str[1]
            df['stim'] = df.index.str.split('^').str[2]

            expr = (
                model.estimates['mean']/
                model.adata.uns['memento']['umi_depth']*
                model.estimates['total_umi'].values).round()

            cov_df = pd.get_dummies(df[['mouse']], drop_first=True).astype(float)
            stim_df = (df[['stim']]==stim).astype(float)
            cov_df = sm.add_constant(cov_df)

            glm_result = model.differential_mean(
                covariates=cov_df, 
                treatments=stim_df,
                family='quasiGLM',
                verbose=2,
                n_jobs=5)

            _, glm_result['fdr'] = fdrcorrection(glm_result['pval'])
            glm_result.to_csv(data_path + 'sc_results/{}_{}_quasiGLM.csv'.format(dataset, trial))

            
            
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
    
    # run_memento()
    
    run_t_mwu()
        