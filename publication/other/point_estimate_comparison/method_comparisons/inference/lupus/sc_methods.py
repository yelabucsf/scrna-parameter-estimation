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
    v = naive_v-(1-q)*(X/(sf**2-sf*(1-q)).reshape(-1,1)).mean(axis=0)
    variance_contributions = ((1-q)/sf).reshape(-1,1)*naive_m.reshape(1,-1) + v.reshape(1,-1)
    m = np.average( X, weights=1/variance_contributions, axis=0)
    m[~np.isfinite(m)] = naive_m[~np.isfinite(m)]
    
    # m = (augmented_data/sf.reshape(-1,1)).mean(axis=0)
    # v = (augmented_data/sf.reshape(-1,1)).var(axis=0)
    # v = v-(1-q)*(X/(sf**2-sf*(1-q)).reshape(-1,1)).mean(axis=0)
    
    return m*data.sum(), (v/data.shape[0])*data.sum()**2


for numcells in [10000,50, 100, 150, 200]:

	for trial in range(10):
		
		print('working on', numcells, trial)

		adata = sc.read(data_path + 'T4_vs_cM.single_cell.{}.{}.h5ad'.format(numcells,trial))

		inds = adata.obs.ind.drop_duplicates().tolist()
		cts = adata.obs.cg_cov.drop_duplicates().tolist()

		### Run the t-test

		def safe_fdr(x):
			fdr = np.ones(x.shape[0])
			_, fdr[np.isfinite(x)] = fdrcorrection(x[np.isfinite(x)])
			return fdr

		# GLM approach
		glm_adata = adata.copy()
		dispersions = pd.read_csv(data_path + 'T4_vs_cM.dispersions.{}.{}.csv'.format(200, trial), index_col=0)
		gene_list = dispersions['gene'].tolist()
		dispersions = dispersions['dispersion'].tolist()

		scaled_means = []
		weights = []
		meta = []
		totals = []
		for ind in inds:
			for ct in ['cM', 'T4']:

				data = glm_adata[(glm_adata.obs['ind']==ind) & (glm_adata.obs['cg_cov']==ct)].X.toarray()
				totals.append(data.sum())
				s, se2 = scaled_mean_se2(data)
				scaled_means.append(s)
				w = np.ones(s.shape[0])
				w[se2>0] = 1/se2[se2>0]
				weights.append(np.sqrt(1/se2))
				meta.append((ind, int(ct=='T4')))
		scaled_means = pd.DataFrame(np.vstack(scaled_means), columns=glm_adata.var.index)
		weights = pd.DataFrame(np.vstack(weights), columns=glm_adata.var.index)
		totals = scaled_means.sum(axis=1).values
		meta = pd.DataFrame(meta, columns=['ind', 'ct'])

		# Filter and re-order by gene_list
		scaled_means = scaled_means[gene_list]
		weights = weights[gene_list]

		weights = weights / weights.values.mean()
		design = dmatrix('ct+ind', meta)


		weighted_mean_glm_results = []
		for idx in range(len(gene_list)):
			model = sm.GLM(
				scaled_means.iloc[:, [idx]], 
				design , 
				exposure=totals,
				var_weights=weights.iloc[:, idx],
				family=sm.families.NegativeBinomial(alpha=np.mean(dispersions)))
			res_model = sm.GLM(
				scaled_means.iloc[:, [idx]], design[:, :-1] , 
				exposure=totals,
				var_weights=weights.iloc[:, idx],
				family=sm.families.NegativeBinomial(alpha=np.mean(dispersions)))
			fit = model.fit()
			res_fit = res_model.fit()
			pv = stats.chi2.sf(-2*(res_fit.llf - fit.llf), df=res_fit.df_resid-fit.df_resid)
			weighted_mean_glm_results.append((fit.params[-1], pv))
		weighted_mean_glm_results = pd.DataFrame(weighted_mean_glm_results, columns=['logFC', 'PValue'], index=gene_list)
		_, weighted_mean_glm_results['FDR'] = fdrcorrection(weighted_mean_glm_results['PValue'])

		weighted_mean_glm_results.to_csv(data_path + 'T4_vs_cM.sc.weighted_mean_glm.{}.{}.csv'.format(numcells, trial))
		
	break