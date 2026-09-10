import scanpy as sc
import scipy as sp
import numpy as np
import pickle as pkl

import sys
sys.path.append('/data/home/Github/scrna-parameter-estimation/scmemo')
import estimator, simulate, scmemo, bootstrap, util, hypothesis_test


if __name__ == '__main__':
	
	data_path = '/data/parameter_estimation/'
	cts = ['CD4 T cells', 'CD14+ Monocytes', 'FCGR3A+ Monocytes', 'NK cells','CD8 T cells', 'B cells']
	label_converter = dict(zip(cts, ['Th', 'cM', 'ncM', 'NK', 'Tc', 'B']))
	
	adata = sc.read(data_path + 'interferon.h5ad')
	adata = adata[(adata.obs.multiplets == 'singlet') & (adata.obs.cell != 'nan'), :].copy()
	adata.X = adata.X.astype(np.int64)
	adata = adata[adata.obs.stim == 'ctrl'].copy()
	
	with open(data_path + 'all_highcount_tfs.pkl', 'rb') as f:
		tfs = pkl.load(f)
		
	for ct in ['CD4 T cells']:#cts:
		
		print('Starting ct', ct)
				
		adata_ct =  adata.copy()
		adata_ct.obs['ct'] = adata_ct.obs['cell'].apply(lambda x: int(x == ct))
		scmemo.create_groups(adata_ct, label_columns=['ct', 'ind'], inplace=True)
		scmemo.compute_1d_moments(
			adata_ct, inplace=True, filter_genes=True, 
			residual_var=True, use_n_umi=False, filter_mean_thresh=0.07, 
			min_perc_group=0.9)
		print('Size of data', adata_ct.shape)
		
		available_tfs = list(set(tfs) & set(adata_ct.var.index.tolist()))
		target_genes = adata_ct.var.index.tolist()
		print('TF list length', len(available_tfs))

		scmemo.compute_2d_moments(adata_ct, available_tfs, target_genes)
			
		scmemo.ht_2d_moments(adata_ct, formula_like='1 + ct', cov_column='ct')
				
		adata_ct.write(data_path + 'result_2d/{}_ct.h5ad'.format(label_converter[ct]))