import pandas as pd
import matplotlib.pyplot as plt
import scanpy as sc
import scipy as sp
import itertools
import numpy as np
import scipy.stats as stats
from scipy.integrate import dblquad
import seaborn as sns
from statsmodels.stats.multitest import fdrcorrection
import statsmodels.api as sm
import pickle as pkl
import time
import string
import random
from sklearn.datasets import make_spd_matrix
from statsmodels.stats.moment_helpers import cov2corr

import sys
sys.path.append('/home/ssm-user/Github/scrna-parameter-estimation/dist/memento-0.1.0-py3.10.egg')
import memento
import memento.simulate as simulate

data_path = '/data_volume/memento/simulation/'
num_replicates = 5


def get_random_string(length):
    # choose from all lowercase letter
    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for i in range(length))
    return result_str


def convert_params_nb(mu, theta):
    """
    Convert mean/dispersion parameterization of a negative binomial to the ones scipy supports

    See https://en.wikipedia.org/wiki/Negative_binomial_distribution#Alternative_formulations
    """
    r = theta
    var = mu + 1 / r * mu ** 2
    p = (var - mu) / var
    return r, 1 - p


def convert_params_binom(mu, v):
    """
    Convert mean/dispersion parameterization of a negative binomial to the ones scipy supports

    See https://en.wikipedia.org/wiki/Negative_binomial_distribution#Alternative_formulations
    """
    p = 1-(v/mu)
    n = mu/p

    return np.ceil(n).astype(int), p


def simulate_counts(means, dispersions, num_cell_per_group):
    
    num_groups, num_genes = means.shape
    
    # counts = np.zeros((num_cells_per_group, num_groups, num_genes))
    
    # binom_idxs = np.where(means >= variances)
    # nb_idxs = np.where(means < variances)
        
    # binom_variances = variances
    # binom_variances[binom_variances > means] = means[binom_variances > means]-1e-3
    # nb_dispersions = (variances-means)/means**2
    # nb_dispersions[nb_dispersions < 0] = 1e-3
    
    # binom_counts = stats.binom.rvs(
    #     *convert_params_binom(means, binom_variances), 
    #     size=(num_cells_per_group, num_groups, num_genes))
    counts = stats.nbinom.rvs(
        *convert_params_nb(means, 1/dispersions), 
        size=(num_cells_per_group, num_groups, num_genes))

    # counts[:, binom_idxs[0], binom_idxs[1]] = binom_counts[:, binom_idxs[0], binom_idxs[1]]
    # counts[:, nb_idxs[0], nb_idxs[1]] = nb_counts[:, nb_idxs[0], nb_idxs[1]]
    
    counts = counts.reshape(-1, num_genes)

    return counts


if __name__ == '__main__':
    ifn_adata = sc.read('/data_volume/memento/hbec/' + 'HBEC_type_I_filtered_counts_deep.h5ad')
    q=0.07
    
    adata_1 = ifn_adata[(ifn_adata.obs['cell_type'] == 'ciliated') & (ifn_adata.obs['stim'] == 'control')]
    adata_2 = ifn_adata[(ifn_adata.obs['cell_type'] == 'ciliated') & (ifn_adata.obs['stim'] == 'beta')]

    x_param_1, z_param_1, Nc_1, good_idx_1 = simulate.extract_parameters(adata_1.X, q=q)
    x_param_2, z_param_2, Nc_2, good_idx_2 = simulate.extract_parameters(adata_2.X, q=q)
    common_set = np.array(list(set(good_idx_1) & set(good_idx_2)))
    z_param_1 = (
        np.array([x for x,i in zip(z_param_1[0], good_idx_1) if i in common_set]),
        np.array([x for x,i in zip(z_param_1[1], good_idx_1) if i in common_set]))
    z_param_2 = (
        np.array([x for x,i in zip(z_param_2[0], good_idx_2) if i in common_set]),
        np.array([x for x,i in zip(z_param_2[1], good_idx_2) if i in common_set]))
    
    pos_var_condition = (z_param_1[1] > 0) & (z_param_2[1] > 0)
    z_param_1 = (z_param_1[0][pos_var_condition], z_param_1[1][pos_var_condition])
    z_param_2 = (z_param_2[0][pos_var_condition], z_param_2[1][pos_var_condition])
    
    estimated_TE = np.log(z_param_2[0]) - np.log(z_param_1[0])
    estimated_TE[np.absolute(estimated_TE) < 0.25] = 0
    available_de_idxs = np.where(estimated_TE > 0)[0]
    
    num_genes = z_param_1[0].shape[0]
    treatment_effect = np.ones(num_genes)
    num_de = 2000
    de_idxs = np.random.choice(available_de_idxs, num_de)
    treatment_effect[de_idxs] = estimated_TE[de_idxs]
    
    conditions = ['ctrl', 'stim']
    groups = [get_random_string(5) for i in range(num_replicates)]
    df = pd.DataFrame(
        itertools.product(groups, conditions),
        columns=['group', 'condition'])

    cov_df = pd.get_dummies(df[['group']], drop_first=True).astype(float)
    cov_df -= cov_df.mean()
    stim_df = (df[['condition']]=='stim').astype(float)
    interaction_df = cov_df*stim_df[['condition']].values
    interaction_df.columns=[f'interaction_{col}' for col in cov_df.columns]
    cov_df = pd.concat([cov_df, interaction_df], axis=1)
    cov_df = sm.add_constant(cov_df)
    design = pd.concat([cov_df, stim_df], axis=1)
        
    base_mean = np.log(z_param_1[0])
    
    mean_beta = np.vstack([
        np.log(z_param_1[0]),
        np.vstack([stats.norm.rvs(scale=0.1, size=num_genes) for i in range((num_replicates-1))]), # intercept random effect
        np.vstack([stats.norm.rvs(scale=0.1, size=num_genes) for i in range((num_replicates-1))]), # treatment random effect
        treatment_effect])
    
    means = np.exp(design.values@mean_beta)
#     base_random = stats.norm.rvs(scale=1, size=num_genes)
#     treatment_random = stats.norm.rvs(scale=1, size=num_genes)
#     de_mask = np.zeros(treatment_random.shape).astype(bool)
#     de_mask[de_idxs] = True
#     treatment_random[~de_mask] = 0
#     multiplier = 0.5
    
#     means[0] = np.exp(base_mean - multiplier*base_random)
#     means[1] = np.exp(base_mean - multiplier*base_random + treatment_effect - multiplier*treatment_random)
#     means[2] = np.exp(base_mean + multiplier*base_random)
#     means[3] = np.exp(base_mean + multiplier*base_random + treatment_effect + multiplier*treatment_random)
    
    # dispersions = np.zeros((num_replicates*2, num_genes))
    dispersions = stats.uniform.rvs(0.1, 1, size=(num_replicates*2, num_genes))
    
    num_cells_per_group = 200
    design = df
    design = pd.concat([design for i in range(num_cells_per_group)])
    
    counts = simulate_counts(means, dispersions, num_cells_per_group).astype(int)
        
    cell_names = [f'cell_{i}' for i in range(counts.shape[0])]
    gene_names = [f'gene_{i}' for i in range(counts.shape[1])]
    design.index=cell_names
    
    _, hyper_captured = simulate.capture_sampling(counts, q=q, process='hyper')

    expr_df = pd.DataFrame(hyper_captured, index=cell_names, columns=gene_names)
    grouped = pd.concat([design, expr_df], axis=1)
    pseudobulks = grouped.groupby(['group', 'condition'])[gene_names].sum()
    pseudobulks.index = [x+'-'+y for x, y in itertools.product(design['group'].unique(), design['condition'].unique())]

    anndata = sc.AnnData(
        sp.sparse.csr_matrix(hyper_captured), 
        obs=design, 
        var=pd.DataFrame(index=gene_names))
    de_genes = np.zeros(num_genes)
    de_genes[de_idxs] = 1
    anndata.var['is_de'] = de_genes.astype(bool)

    anndata.write(data_path + 'de/anndata.h5ad')
    pseudobulks.T.to_csv(data_path + 'de/pseudobulks.csv')
    
    norm_adata = anndata.copy().copy()
    sc.pp.normalize_total(norm_adata, target_sum=1)
    norm_adata.write(data_path + 'de/norm_anndata.h5ad')
    
    print('data generation done')