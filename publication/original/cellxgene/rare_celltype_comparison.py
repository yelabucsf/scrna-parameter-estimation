import tiledb
import tiledbsoma as soma
from somacore import ExperimentAxisQuery, AxisQuery
import pandas as pd
pd.set_option('display.max_columns', 200)
import numpy as np
import itertools
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt
import time
import scanpy as sc
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from pymare import meta_regression
from functools import partial
from joblib import Parallel, delayed


CUBE_PATH = '/home/ubuntu/Github/memento-cxg/'
SAVE_PATH = '/home/ubuntu/Data/mementocxg/'

CUBE_TILEDB_DIMS_OBS = [
    "cell_type",
    "dataset_id",
]

CUBE_TILEDB_ATTRS_OBS = [
    "assay",
    "suspension_type",
    "donor_id",
    "disease",
    "sex"
]

CUBE_LOGICAL_DIMS_OBS = CUBE_TILEDB_DIMS_OBS + CUBE_TILEDB_ATTRS_OBS

DE_TREATMENT = 'treatment'
DE_COVARIATES = ['donor_id']
DE_VARIABLES = [DE_TREATMENT] + DE_COVARIATES

LFC_THRESHOLD = 0

DATASETS = [
    '2672b679-8048-4f5e-9786-f1b196ccfd08',
    '86282760-5099-4c71-8cdd-412dcbbbd0b9',
    '2872f4b0-b171-46e2-abc6-befcf6de6306',
    '644a578d-ffdc-446b-9679-e7ab4c919c13',
    '11ff73e8-d3e4-4445-9309-477a2c5be6f6',
    '4dd00779-7f73-4f50-89bb-e2d3c6b71b18',
    'bd65a70f-b274-4133-b9dd-0d1431b6af34',
    'b07fb54c-d7ad-4995-8bb0-8f3d8611cabe',
    '3f32121d-126b-4e8d-9f69-d86502d2a1b1',
    'a51c6ece-5731-4128-8c1e-5060e80c69e4',
    'd7dcfd8f-2ee7-4385-b9ac-e074c23ed190',
    '2d31c0ca-0233-41ce-bd1a-05aa8404b073',
    '1e5bd3b8-6a0e-4959-8d69-cafed30fe814',
    # 'cd4c96bb-ad66-4e83-ba9e-a7df8790eb12',
    '44882825-0da1-4547-b721-2c6105d4a9d1',
    '4ed927e9-c099-49af-b8ce-a2652d069333',
    '00ff600e-6e2e-4d76-846f-0eec4f0ae417',
    '105c7dad-0468-4628-a5be-2bb42c6a8ae4',
    'c5d88abe-f23a-45fa-a534-788985e93dad',
    # 'ed5d841d-6346-47d4-ab2f-7119ad7e3a35',
    '53d208b0-2cfd-4366-9866-c3c6114081bc',
    '3de0ad6d-4378-4f62-b37b-ec0b75a50d94',
    '574e9f9e-f8b4-41ef-bf19-89a9964fd9c7',
    '1a2e3350-28a8-4f49-b33c-5b67ceb001f6',
    'd3a83885-5198-4b04-8314-b753b66ef9a8']

def treatment_assignment(row):

    ct = row['cell_type']
    if 'plasma' in ct:
        return 'pdc'
    if 'conven' in ct or 'myeloid' in ct:
        return 'cdc'
    else:
        return 'unknown'
    

def get_cell_counts():
    
    dataset_ids_to_query = DATASETS
    dataset_query = '('
    for idx, di in enumerate(dataset_ids_to_query): 
        dataset_query += f'dataset_id == "{di}" '
        if idx != len(dataset_ids_to_query)-1:
            dataset_query += 'or '
    dataset_query += ')'

    celltypes_to_query = [
        'conventional dendritic cell',
        'plasmacytoid dendritic cell',
        'conventional dendritic cell',
        'plasmacytoid dendritic cell, human',
        'dendritic cell',
        'dendritic cell, human',
        'myeloid dendritic cell',
        'plasmacytoid dendritic cell']
    celltype_query = '('
    for idx, ct in enumerate(celltypes_to_query): 
        celltype_query += f'cell_type == "{ct}" '
        if idx != len(celltypes_to_query)-1:
            celltype_query += 'or '
    celltype_query += ')'

    OBS_VALUE_FILTER_1 = dataset_query # All cells in three datasets
    OBS_VALUE_FILTER_2 = dataset_query + ' and ' + celltype_query # only relevant celltypes

    exp_uri = 's3://cellxgene-data-public/cell-census/2023-10-30/soma/census_data/homo_sapiens'
    layer = "raw"
    measurement_name = "RNA"

    with soma.Experiment.open(uri=exp_uri,
                              context=soma.SOMATileDBContext().replace(tiledb_config={
                                  "vfs.s3.region":"us-west-2",
                                  "vfs.s3.no_sign_request":True})
                              ) as exp:

        query = exp.axis_query(measurement_name=measurement_name,
                               obs_query=AxisQuery(value_filter=OBS_VALUE_FILTER_2),
                               # Note: Must use *all* genes to compute size factors correctly, even when var filter is
                               # being used for testing
                               var_query=AxisQuery())
    obs_df = query.obs().concat().to_pandas()
    
    cell_count = obs_df.groupby(['cell_type', 'donor_id']).size().reset_index(name='count')
    cell_count['treatment'] = cell_count.apply(treatment_assignment, axis=1)

    names = cell_count[DE_TREATMENT].copy()
    for col in DE_COVARIATES:
        names += '_' + cell_count[col]
    cell_count['group_name'] = names.tolist()
    cell_count = cell_count.drop_duplicates('group_name').set_index('group_name')
    
    return cell_count

    
def get_final_design_matrix(design):
    cov_df = design.iloc[:, 1:]
    cov_df -= cov_df.mean(axis=0)
    stim_df = design.iloc[:, [0]]
    interaction_df = cov_df*stim_df[['treatment']].values
    interaction_df.columns=[f'interaction_{col}' for col in cov_df.columns]
    cov_df = pd.concat([cov_df, interaction_df], axis=1)
    cov_df = sm.add_constant(cov_df)
    return  pd.concat([stim_df, cov_df], axis=1).values.astype(float)


def wls(X, y, n, v, tau2=0, thresh=1):

    from sklearn.linear_model import LinearRegression
    

    # fit WLS using sample_weights
    WLS = LinearRegression(fit_intercept=False)
    WLS.fit(X, y)
    sample_err = ((WLS.predict(X) - y)**2).mean()
    # print(WLS.coef_)

    coef = WLS.coef_[0]

    # W = np.diag(1/ (v) )
    W = 1/v
    try:
        beta_var_hat = np.diag(np.linalg.pinv((X.T *W)@X ))
    except:
        return np.nan, np.nan, 0, 0
    # print(beta_var_hat)
    se = np.sqrt( beta_var_hat[0] )
    
    if np.abs(coef) < LFC_THRESHOLD:
        z = 0
        pv = 1
    else:
        z = (np.abs(coef)-LFC_THRESHOLD)/se
        pv = stats.norm.sf(np.abs(z))*2

    return coef, se, z, pv


def compare_rare_cell_types(all_estimators):
    
    all_estimators['treatment'] = all_estimators.apply(treatment_assignment, axis=1)

    estimators = all_estimators.query('treatment != "unknown"').copy()

    donors_to_use = estimators[['cell_type', 'donor_id']].drop_duplicates().groupby('donor_id').size()
    donors_to_use = donors_to_use[donors_to_use > 1].index.tolist()
    estimators = estimators.query('donor_id in @donors_to_use').copy()
    
    names = estimators[DE_TREATMENT].copy()
    for col in DE_COVARIATES:
        names += '_' + estimators[col]
    estimators['group_name'] = names.tolist()

    estimators = estimators.drop_duplicates(subset=['group_name', 'feature_id'])

    features = estimators['feature_id'].drop_duplicates().tolist()
    
    groups = estimators.drop_duplicates(subset='group_name').set_index('group_name')
    
    mean = estimators.pivot(index='group_name', columns='feature_id', values='mean')
    se_mean = estimators.pivot(index='group_name', columns='feature_id', values='sem')
    cell_counts = get_cell_counts()
    
    # Filter genes for actually expressed ones
    genes_to_test = mean.columns[mean.isnull().values.mean(axis=0) < 0.7]
    mean = mean.loc[groups.index, genes_to_test]
    se_mean = se_mean.loc[groups.index, genes_to_test]
    cell_counts = cell_counts.loc[groups.index]
    
    design = groups[DE_VARIABLES].copy()
    design['treatment'] = (design['treatment'] == 'pdc').astype(float)
    design['constant'] = 1
    
    regression_de_result = []
    
    def run_gene_de(feature, m, sem, counts, design):
        
        # Transform to log space (alternatively can resample in log space)
        lm = np.log(m)
        selm = (np.log(m+sem)-np.log(m-sem))/2
        sample_idxs = np.isfinite(m) & np.isfinite(sem) & (counts > 20)
        sample_design = design[['treatment', 'donor_id']].iloc[sample_idxs]

        donors_to_use_ = sample_design.groupby('donor_id').size()
        donors_to_use_ = donors_to_use_[donors_to_use_ > 1].index.tolist()

        final_sample_idxs = sample_design['donor_id'].isin(donors_to_use_).values
        
        if final_sample_idxs.sum() < 2:
            return feature, np.nan, np.nan, np.nan, np.nan

        # final_sample_idxs = np.ones(final_sample_idxs.shape[0]).astype(bool)

        X, y, n, v = (
            get_final_design_matrix(pd.get_dummies(sample_design.iloc[final_sample_idxs], columns=['donor_id'], drop_first=True)), 
            lm[sample_idxs][final_sample_idxs], 
            cell_counts[sample_idxs][final_sample_idxs], 
            selm[sample_idxs][final_sample_idxs]**2)
        
        coef, se, z, pv = wls(X, y, n, v)
        
        return feature, coef, se, z, pv
    
    tasks = []
    for feature in genes_to_test: # Can be vectorized heavily, showing for 1K genes

        m = mean[feature].values
        sem = se_mean[feature].values

        tasks.append(partial(run_gene_de, feature, m, sem, cell_counts['count'].values, design))
    
    regression_de_result = Parallel(n_jobs=30, verbose=5)(delayed(task)() for task in tasks)
    regression_de_result_wls = pd.DataFrame(regression_de_result, columns=['feature_id','coef', 'se','z', 'pval']).set_index('feature_id')
        
    return regression_de_result_wls
    
    
if __name__ == '__main__':
    
    
    
    cube = tiledb.open(CUBE_PATH + 'estimators_cube_dcs_many').df[:]\
        .rename(columns={
        'feature_id':'cell_type',
        'cell_type':'dataset_id',
        'dataset_id':'feature_id',
    })
    
    # Entire dataset
    results = compare_rare_cell_types(cube.query('dataset_id in @DATASETS'))
    results.to_csv(SAVE_PATH + 'rare_ct_whole.csv')
    
    # # Using 1 dataset each
    # for dataset_id in DATASETS:
    #     results = compare_rare_cell_types(cube.query(f'dataset_id == "{dataset_id}"').copy())
    #     results.to_csv(SAVE_PATH + f'rare_ct_{dataset_id}.csv')
