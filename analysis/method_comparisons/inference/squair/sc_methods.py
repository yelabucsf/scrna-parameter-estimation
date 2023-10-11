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

data_path = '/data_volume/memento/method_comparison/lupus/'

columns = ['logFC', 'PValue', 'FDR']

def sample_sum(data):
    
    s = data.sum(axis=0)
    return s

def scaled_mean_se2(data):
    
    augmented_data = np.append(data, np.ones((1,data.shape[1])), axis=0)

    q=0.07
    sf = augmented_data.sum(axis=1)
    X = augmented_data/sf.reshape(-1,1)
    
    naive_v = X.var(axis=0)
    naive_m = X.mean(axis=0)
    v = naive_v#-(1-q)*(X/(sf**2-sf*(1-q)).reshape(-1,1)).mean(axis=0)
    variance_contributions = ((1-q)/sf).reshape(-1,1)*naive_m.reshape(1,-1) + v.reshape(1,-1)
    m = np.average( X, weights=1/variance_contributions, axis=0)
    m[~np.isfinite(m)] = naive_m[~np.isfinite(m)]
    
    # m = (augmented_data/sf.reshape(-1,1)).mean(axis=0)
    # v = (augmented_data/sf.reshape(-1,1)).var(axis=0)
    # v = v-(1-q)*(X/(sf**2-sf*(1-q)).reshape(-1,1)).mean(axis=0)
    
    return m*data.sum(), (v/data.shape[0])*data.sum()**2

for numcells in [50, 100, 150, 200]:

	for trial in range(50):
		
		print('working on', numcells, trial)

		adata = sc.read(data_path + 'T4_vs_cM.single_cell.{}.{}.h5ad'.format(numcells,trial))

		inds = adata.obs.ind.drop_duplicates().tolist()
		cts = adata.obs.cg_cov.drop_duplicates().tolist()

		### Run the t-test

		def safe_fdr(x):
			fdr = np.ones(x.shape[0])
			_, fdr[np.isfinite(x)] = fdrcorrection(x[np.isfinite(x)])
			return fdr

		ttest_adata = adata.copy()
		sc.pp.normalize_total(ttest_adata)
		sc.pp.log1p(ttest_adata)

		data1 = ttest_adata[ttest_adata.obs['cg_cov'] =='T4'].X.todense()
		data2 = ttest_adata[ttest_adata.obs['cg_cov'] =='cM'].X.todense()

		statistic, pvalue = stats.ttest_ind(data1, data2, axis=0)

		logfc = data1.mean(axis=0) - data2.mean(axis=0)

		ttest_result = pd.DataFrame(
			zip(logfc.A1, pvalue, safe_fdr(pvalue)), 
			index=ttest_adata.var.index,
			columns=columns)
		ttest_result.to_csv(data_path + 'T4_vs_cM.sc.ttest.{}.{}.csv'.format(numcells, trial))

		mwu_stat, mwu_pval = stats.mannwhitneyu(data1, data2, axis=0)
		mwu_result = pd.DataFrame(
			zip(logfc.A1, mwu_pval, safe_fdr(mwu_pval)), 
			index=ttest_adata.var.index,
			columns=columns)
		mwu_result.to_csv(data_path + 'T4_vs_cM.sc.mwu.{}.{}.csv'.format(numcells, trial))