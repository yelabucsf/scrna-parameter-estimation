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


logging.basicConfig(
    format="%(asctime)s %(process)-7s %(levelname)-8s %(message)s",
    level=logging.INFO, 
    datefmt="%Y-%m-%d %H:%M:%S",
)
logging.captureWarnings(True)

data_path = '/data_volume/bulkrna/lupus/'


datasets = [
 'CD4_Memory-Th0',
 'CD4_Memory-Th2',
 'CD4_Memory-Th17',
 'CD4_Memory-iTreg',
 'CD4_Naive-Th0',
 'CD4_Naive-Th2',
 'CD4_Naive-Th17',
 'CD4_Naive-iTreg']


def safe_fdr(x):
    fdr = np.ones(x.shape[0])
    _, fdr[np.isfinite(x)] = fdrcorrection(x[np.isfinite(x)])
    return fdr
    
    
if __name__ == '__main__':
    
    columns = ['logFC','PValue', 'FDR']
    
    numcells = 100

    for trial in range(50):

        print('working on', numcells, trial)

        adata = sc.read(data_path + 'T4_vs_cM.single_cell.{}.{}.h5ad'.format(numcells,trial))

        data1 = adata[adata.obs['cg_cov'] =='T4'].X.todense()
        data2 = adata[adata.obs['cg_cov'] =='cM'].X.todense()

        statistic, pvalue = stats.ttest_ind(data1, data2, axis=0)

        logfc = data1.mean(axis=0) - data2.mean(axis=0)

        ttest_result = pd.DataFrame(
            zip(logfc.A1, pvalue, safe_fdr(pvalue)), 
            index=adata.var.index,
            columns=columns)
        ttest_result.to_csv(data_path + f'{numcells}_{trial}_t.csv')

        mwu_stat, mwu_pval = stats.mannwhitneyu(data1, data2, axis=0)
        mwu_result = pd.DataFrame(
            zip(logfc.A1, mwu_pval, safe_fdr(mwu_pval)), 
            index=adata.var.index,
            columns=columns)
        mwu_result.to_csv(data_path + f'{numcells}_{trial}_mwu.csv')
        