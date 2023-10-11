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

data_path = '/data_volume/memento/lupus_bulk/'


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
    
    for numcells in [500]:#[100, 10000]:

        for trial in range(50):

            print('working on', numcells, trial)

            adata = sc.read(data_path + 'T4_vs_cM.single_cell.{}.{}.h5ad'.format(numcells,trial))

            adata.obs['q'] = 0.07
            adata.X = adata.X.astype(float)

            rna.MementoRNA.setup_anndata(
                    adata=adata,
                    q_column='q',
                    label_columns=['ind', 'cg_cov'],
                    num_bins=30)

            adata = adata[:, adata.X.mean(axis=0).A1 > 0.02]

            model = rna.MementoRNA(adata=adata)

            model.compute_estimate(
                estimand='mean',
                get_se=True,
                n_jobs=30,
            )

            df = pd.DataFrame(index=adata.uns['memento']['groups'])
            df['ind'] = df.index.str.split('^').str[1]
            df['ct'] = df.index.str.split('^').str[2]

            cov_df = pd.get_dummies(df[['ind']], drop_first=True).astype(float)
            stim_df = (df[['ct']]=='T4').astype(float)
            cov_df = sm.add_constant(cov_df)

            glm_result = model.differential_mean(
                covariates=cov_df, 
                treatments=stim_df,
                family='quasiGLM',
                verbose=2,
                n_jobs=5)

            _, glm_result['fdr'] = fdrcorrection(glm_result['pval'])
            glm_result.to_csv(data_path + '{}_{}_quasiGLM.csv'.format(numcells, trial))
        