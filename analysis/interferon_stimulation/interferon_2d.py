import scanpy as sc
import scipy as sp
import numpy as np
import pickle as pkl

import sys
sys.path.append('/home/ssm-user/Github/scrna-parameter-estimation/scmemo')
import estimator, simulate, scmemo, bootstrap, util, hypothesis_test


if __name__ == '__main__':
	
	data_path = '/data/'
	cts = ['CD4 T cells', 'CD14+ Monocytes', 'FCGR3A+ Monocytes', 'NK cells','CD8 T cells', 'B cells']
	label_converter = dict(zip(cts, ['Th', 'cM', 'ncM', 'NK', 'Tc', 'B']))
	
	adata = sc.read(data_path + 'interferon.h5ad')
	adata = adata[(adata.obs.multiplets == 'singlet') & (adata.obs.cell != 'nan'), :].copy()
	adata.X = adata.X.astype(np.int64)
	
	with open(data_path + 'all_highcount_tfs.pkl', 'rb') as f:
		tfs = pkl.load(f)
		
	for ct in cts[:3]:
		
		print('Starting ct', ct)
				
		adata_ct =  adata[adata.obs.cell == ct].copy()
		scmemo.create_groups(adata_ct, label_columns=['stim', 'ind'], inplace=True)
		scmemo.compute_1d_moments(
			adata_ct, inplace=True, filter_genes=True, 
			residual_var=True, use_n_umi=False, filter_mean_thresh=0.07, 
			min_perc_group=0.8)
		print('Size of data', adata_ct.shape)
		
		available_tfs = list(set(tfs) & set(adata_ct.var.index.tolist()))
		target_genes = adata_ct.var.index.tolist()
		print('TF list length', len(available_tfs))

		scmemo.compute_2d_moments(adata_ct, available_tfs, target_genes)
			
		scmemo.ht_2d_moments(adata_ct, formula_like='1 + stim', cov_column='stim', num_cpu=11)
				
		adata_ct.write(data_path + 'result_2d/{}_2d_result_2.h5ad'.format(label_converter[ct]))