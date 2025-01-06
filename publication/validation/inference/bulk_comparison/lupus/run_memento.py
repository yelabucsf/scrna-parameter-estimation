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

    
if __name__ == '__main__':
    
    numcells = 100

    for trial in range(50):

        print('working on', numcells, trial)

        adata = sc.read(data_path + 'T4_vs_cM.single_cell.{}.{}.h5ad'.format(numcells,trial))

        adata.obs['q'] = 0.07
        memento.setup_memento(
            adata, 
            'q', 
            filter_mean_thresh=0.00, 
            estimator_type='pseudobulk')

        memento.create_groups(adata, label_columns=['ind', 'cg_cov'])
        gene_list = adata.var.index[adata.X.mean(axis=0).A1 > 0.02].tolist()

        memento.compute_1d_moments(adata, min_perc_group=0, gene_list=gene_list)

        df = pd.DataFrame(index=adata.uns['memento']['groups'])
        df['ind'] = df.index.str.split('^').str[1]
        df['ct'] = df.index.str.split('^').str[2]

        cov_df = pd.get_dummies(df[['ind']], drop_first=True).astype(float)
        stim_df = (df[['ct']]=='T4').astype(float)
        cov_df = sm.add_constant(cov_df)[['const']]

        memento.main.ht_mean(
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
        results.to_csv(data_path + '{}_{}_quasiML.csv'.format(numcells, trial))
        