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

import sys
sys.path.append('/home/ssm-user/Github/scrna-parameter-estimation/dist/memento-0.1.0-py3.8.egg')
sys.path.append('/home/ssm-user/Github/misc-seq/miscseq/')
import encode
import memento

data_path = '/data_volume/memento/method_comparison/hbec/'

columns = ['logFC', 'PValue', 'FDR']

def sample_sum(data):
    
    s = data.sum(axis=0)
    return s

def scaled_mean_se2(data, sf, q):

	augmented_data = np.append(data, np.ones((1,data.shape[1])), axis=0)

	sf = np.append(sf, sf.mean())
	q = q.mean()
	X = augmented_data/sf.reshape(-1,1)

	naive_v = X.var(axis=0)
	naive_m = X.mean(axis=0)
	v = naive_v-(1-q)*(X/(sf**2-sf*(1-q)).reshape(-1,1)).mean(axis=0)
	variance_contributions = ((1-q)/sf).reshape(-1,1)*naive_m.reshape(1,-1) + v.reshape(1,-1)
	m = np.average( X, weights=1/variance_contributions, axis=0)
	m[~np.isfinite(m)] = naive_m[~np.isfinite(m)]
	m[m<0] = 0
	# return m, v/data.shape[0]
	return m*data.sum(), (v/data.shape[0])*data.sum()**2

def assign_q(batch):

	if batch == 0:
		return 0.387*0.25
	elif batch == 1:
		return 0.392*0.25
	elif batch == 2:
		return 0.436*0.25
	else:
		return 0.417*0.25
	
def run_sc_methods():

	for ct in ['BC', 'B', 'C']:

		for tp in ['3', '6', '9', '24', '48']:

			for stim in ['alpha', 'beta']:

				print('working on', ct, tp, stim)

				adata = sc.read(data_path + 'hbec.single_cell.{}.{}.{}.h5ad'.format(ct, tp, stim))
				adata.obs['q'] = adata.obs['batch'].apply(assign_q)
				memento.setup_memento(adata, q_column='q', trim_percent=0.05)

				inds = adata.obs.donor.drop_duplicates().tolist()
				conditions = adata.obs.stim.drop_duplicates().tolist()

				### Run the t-test

				def safe_fdr(x):
					fdr = np.ones(x.shape[0])
					_, fdr[np.isfinite(x)] = fdrcorrection(x[np.isfinite(x)])
					return fdr

				ttest_adata = adata.copy()
				sc.pp.normalize_total(ttest_adata)
				sc.pp.log1p(ttest_adata)

				data1 = ttest_adata[ttest_adata.obs['stim'] =='control'].X.todense()
				data2 = ttest_adata[ttest_adata.obs['stim'] ==stim].X.todense()

				statistic, pvalue = stats.ttest_ind(data1, data2, axis=0)

				logfc = data1.mean(axis=0) - data2.mean(axis=0)

				ttest_result = pd.DataFrame(
					zip(logfc.A1, pvalue, safe_fdr(pvalue)), 
					index=ttest_adata.var.index,
					columns=columns)
				ttest_result.to_csv(data_path + 'hbec.sc.ttest.{}.{}.{}.csv'.format(ct, tp, stim))

				mwu_stat, mwu_pval = stats.mannwhitneyu(data1, data2, axis=0)
				mwu_result = pd.DataFrame(
					zip(logfc.A1, mwu_pval, safe_fdr(mwu_pval)), 
					index=ttest_adata.var.index,
					columns=columns)
				mwu_result.to_csv(data_path + 'hbec.sc.mwu.{}.{}.{}.csv'.format(ct, tp, stim))
				
				# GLM approach
				glm_adata = adata.copy()
				dispersions = pd.read_csv(data_path + 'hbec.dispersions.{}.{}.{}.csv'.format(ct, tp, stim), index_col=0)
				gene_list = dispersions['gene'].tolist()
				dispersions = dispersions['dispersion'].tolist()

				scaled_means = []
				weights = []
				meta = []
				totals = []
				for ind in inds:
					for condition in [stim, 'control']:

						data = glm_adata[(glm_adata.obs['donor']==ind) & (glm_adata.obs['stim']==condition)].X.toarray()
						sf = glm_adata[(glm_adata.obs['donor']==ind) & (glm_adata.obs['stim']==condition)].obs['memento_size_factor'].values
						q = glm_adata[(glm_adata.obs['donor']==ind) & (glm_adata.obs['stim']==condition)].obs['q'].values
						totals.append(data.sum())
						s, se2 = scaled_mean_se2(data, sf, q)
						scaled_means.append(s)
						w = np.ones(s.shape[0])
						w[se2>0] = 1/se2[se2>0]
						weights.append(np.sqrt(1/se2))
						meta.append((ind, int(condition==stim)))
				scaled_means = pd.DataFrame(np.vstack(scaled_means), columns=glm_adata.var.index)
				weights = pd.DataFrame(np.vstack(weights), columns=glm_adata.var.index)
				totals = scaled_means.sum(axis=1).values

				meta = pd.DataFrame(meta, columns=['ind', 'stim'])

				# Filter and re-order by gene_list
				scaled_means = scaled_means[gene_list]
				weights = weights[gene_list]
				weights = weights / weights.values.mean(axis=0)

				design = dmatrix('stim+ind', meta)


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

				weighted_mean_glm_results.to_csv(data_path + 'hbec.sc.weighted_mean_glm.{}.{}.{}.csv'.format(ct, tp, stim))

def run_combined_sc_methods():

	for tp in ['48']:

		for stim in ['alpha', 'beta']:

			print('working on', tp, stim)

			adata = sc.read(data_path + 'hbec.single_cell.{}.{}.h5ad'.format(tp, stim))
			adata.obs['q'] = adata.obs['batch'].apply(assign_q)
			memento.setup_memento(adata, q_column='q', trim_percent=0.05)

			inds = adata.obs.donor.drop_duplicates().tolist()
			conditions = adata.obs.stim.drop_duplicates().tolist()

			### Run the t-test

			def safe_fdr(x):
				fdr = np.ones(x.shape[0])
				_, fdr[np.isfinite(x)] = fdrcorrection(x[np.isfinite(x)])
				return fdr

			ttest_adata = adata.copy()
			sc.pp.normalize_total(ttest_adata)
			sc.pp.log1p(ttest_adata)

			data1 = ttest_adata[ttest_adata.obs['stim'] =='control'].X.todense()
			data2 = ttest_adata[ttest_adata.obs['stim'] ==stim].X.todense()

			statistic, pvalue = stats.ttest_ind(data1, data2, axis=0)

			logfc = data1.mean(axis=0) - data2.mean(axis=0)
			
			ttest_result = pd.DataFrame(
				zip(logfc.A1, pvalue, safe_fdr(pvalue)), 
				index=ttest_adata.var.index,
				columns=columns)
			ttest_result.to_csv(data_path + 'hbec.sc.ttest.{}.{}.csv'.format(tp, stim))
			
			# MWU
			mwu_stat, mwu_pval = stats.mannwhitneyu(data1, data2, axis=0)
			mwu_result = pd.DataFrame(
				zip(logfc.A1, mwu_pval, safe_fdr(mwu_pval)), 
				index=ttest_adata.var.index,
				columns=columns)
			mwu_result.to_csv(data_path + 'hbec.sc.mwu.{}.{}.csv'.format(tp, stim))
			
			# GLM approach
			glm_adata = adata.copy()
			dispersions = pd.read_csv(data_path + 'hbec.dispersions.{}.{}.{}.csv'.format('BC', tp, stim), index_col=0)
			gene_list = dispersions['gene'].tolist()
			dispersions = dispersions['dispersion'].tolist()

			scaled_means = []
			weights = []
			meta = []
			totals = []
			for ct in ['BC', 'B', 'C']:
				for ind in inds:
					for condition in [stim, 'control']:

						data = glm_adata[(glm_adata.obs['donor']==ind) & (glm_adata.obs['stim']==condition) & (glm_adata.obs['ct']==ct)].X.toarray()
						sf = glm_adata[(glm_adata.obs['donor']==ind) & (glm_adata.obs['stim']==condition) & (glm_adata.obs['ct']==ct)].obs['memento_size_factor'].values
						q = glm_adata[(glm_adata.obs['donor']==ind) & (glm_adata.obs['stim']==condition) & (glm_adata.obs['ct']==ct)].obs['q'].values
						totals.append(data.sum())
						s, se2 = scaled_mean_se2(data, sf, q)
						scaled_means.append(s)
						w = np.ones(s.shape[0])
						w[se2>0] = 1/se2[se2>0]
						weights.append(np.sqrt(1/se2))
						meta.append((ind, ct, int(condition==stim)))
			scaled_means = pd.DataFrame(np.vstack(scaled_means), columns=glm_adata.var.index)
			weights = pd.DataFrame(np.vstack(weights), columns=glm_adata.var.index)
			totals = scaled_means.sum(axis=1).values

			meta = pd.DataFrame(meta, columns=['ind', 'ct','stim'])

			# Filter and re-order by gene_list
			scaled_means = scaled_means[gene_list]
			weights = weights[gene_list]
			weights = weights / weights.values.mean(axis=0)

			design = dmatrix('stim+ind+ct', meta)


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

			weighted_mean_glm_results.to_csv(data_path + 'hbec.sc.weighted_mean_glm.{}.{}.csv'.format(tp, stim))
			
if __name__ == '__main__':
	
	# run_sc_methods()
	run_combined_sc_methods()