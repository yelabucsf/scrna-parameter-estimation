import pandas as pd
import matplotlib.pyplot as plt
import scanpy as sc
import scipy as sp
import itertools
import numpy as np
import scipy.stats as stats
from scipy.integrate import dblquad
from statsmodels.stats.multitest import fdrcorrection
import random
import statsmodels.api as sm

import sys
sys.path.append('/home/ssm-user/Github/memento')

import memento.model.rna as rna
import memento.estimator.hypergeometric as hg
import memento.util as util

import logging
logging.basicConfig(
    format="%(asctime)s %(process)-7s %(levelname)-8s %(message)s",
    level=logging.INFO, 
    datefmt="%Y-%m-%d %H:%M:%S",
)
logging.captureWarnings(True)


q=0.07
data_path = '/data_volume/memento/simulation/'

if __name__ == '__main__':
    
    adata = sc.read(data_path + 'de/anndata.h5ad')

    adata.obs['q'] = q
    adata.X = adata.X.astype(float)

    rna.MementoRNA.setup_anndata(
            adata=adata,
            q_column='q',
            label_columns=['group', 'condition'],
            num_bins=30,
            trim_percent=0.1,
            shrinkage=0.5)

    adata = adata[:, adata.X.mean(axis=0).A1 > 0.05]
    model = rna.MementoRNA(adata=adata)

    model.compute_estimate(
        estimand='mean',
        get_se=True,
        n_jobs=30,
    )
    
    df = pd.DataFrame(index=adata.uns['memento']['groups'])
    df['group'] = df.index.str.split('^').str[1]
    df['condition'] = df.index.str.split('^').str[2]

    cov_df = pd.get_dummies(df[['group']], drop_first=True).astype(float)
    cov_df -= cov_df.mean()
    stim_df = (df[['condition']]=='stim').astype(float)
    interaction_df = cov_df*stim_df[['condition']].values
    interaction_df.columns=[f'interaction_{col}' for col in cov_df.columns]
    cov_df = pd.concat([cov_df, interaction_df], axis=1)
    cov_df = sm.add_constant(cov_df)

    # glm_result = model.differential_mean(
    #     covariates=cov_df, 
    #     treatments=stim_df,
    #     family='quasiGLM',
    #     verbose=0,
    #     n_jobs=10)

    # _, glm_result['fdr'] = fdrcorrection(glm_result['pval'])
    # glm_result.to_csv(data_path + 'de/memento.csv')
    
    wls_result = model.differential_mean(
        covariates=cov_df, 
        treatments=stim_df,
        family='WLS',
        verbose=0,
        n_jobs=10)

    _, wls_result['fdr'] = fdrcorrection(wls_result['pval'])
    wls_result.to_csv(data_path + 'de/memento_wls.csv')

    wls_result = wls_result.join(adata.var[['is_de']],how='inner')
    
    check_thresh = 0.05
    tpr = (wls_result.query('is_de')['pval'] < check_thresh).mean()
    fpr = (wls_result.query('~is_de')['pval'] < check_thresh).mean()
    
    print(f'memento successful, fpr = {fpr}, tpr = {tpr}')
