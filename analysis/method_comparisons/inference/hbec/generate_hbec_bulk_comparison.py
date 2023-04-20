import pandas as pd
import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt

data_path = '/data_volume/memento/method_comparison/hbec/'
### Read single cell and bulk data

adata = sc.read(data_path + 'HBEC_type_I_filtered_counts_deep.h5ad')
adata = adata[adata.obs.stim.isin(['alpha', 'beta', 'control'])]
converter = {'basal/club':'BC', 'basal':'B', 'ciliated':'C', 'goblet':'G', 'ionocyte/tuft':'IT', 'neuroendo':'N', 'club':'club'}
adata.obs['ct'] = adata.obs['cell_type'].apply(lambda x: converter[x])


def generate_datasets():
	
	inds = list(set(adata.obs.donor))
	for ct in ['BC', 'B', 'C']:

		for tp in ['3', '6', '9', '24', '48']:

			for stim in ['alpha', 'beta']:

				subset = adata[
					(adata.obs['ct'] == ct) & \
					(adata.obs['stim'].isin([stim, 'control'])) & \
					(adata.obs['time'].isin([tp, '0']))
				]
				pseudobulks = []
				names = []
				adata_list = []
				for condition in [stim, 'control']:
					for ind in inds:
						stim_ind_adata = subset[(subset.obs['donor']==ind) & (subset.obs['stim']==condition)].copy()
						print('processing', ct, tp, stim, condition, ind, 'num cells:', stim_ind_adata.shape[0])
						adata_list.append(stim_ind_adata.copy())
						pseudobulks.append( stim_ind_adata.X.sum(axis=0).A1)
						names.append('_'.join([ct, tp, condition, ind]) )
				sc_data = sc.AnnData.concatenate(*adata_list)
				pseudobulks = np.vstack(pseudobulks)
				pseudobulks = pd.DataFrame(pseudobulks.T, columns=names, index=subset.var.index.tolist())

				pseudobulks.to_csv(data_path + 'hbec.pseudobulk.{}.{}.{}.csv'.format(ct,tp, stim))

				sc_data.write(data_path + 'hbec.single_cell.{}.{}.{}.h5ad'.format(ct,tp, stim))

				
def generate_combined_datasets():
	
	inds = list(set(adata.obs.donor))

	for tp in ['3', '6', '9', '24', '48']:

		for stim in ['alpha', 'beta']:
			
			print('making combined ct dataset for', tp, stim)

			subset = adata[
				(adata.obs['stim'].isin([stim, 'control'])) & \
				(adata.obs['time'].isin([tp, '0']))
			]
			pseudobulks = []
			names = []
			adata_list = []
			for condition in [stim, 'control']:
				for ind in inds:
					for ct in ['BC', 'B', 'C']:
						stim_ind_adata = subset[(subset.obs['donor']==ind) & (subset.obs['stim']==condition) & (subset.obs['ct']==ct)].copy()
						print('processing', ct, tp, stim, condition, ind, 'num cells:', stim_ind_adata.shape[0])
						adata_list.append(stim_ind_adata.copy())
						pseudobulks.append( stim_ind_adata.X.sum(axis=0).A1)
						names.append('_'.join([ct, tp, condition, ind]) )
			sc_data = sc.AnnData.concatenate(*adata_list)
			pseudobulks = np.vstack(pseudobulks)
			pseudobulks = pd.DataFrame(pseudobulks.T, columns=names, index=subset.var.index.tolist())

			pseudobulks.to_csv(data_path + 'hbec.pseudobulk.{}.{}.csv'.format(tp, stim))

			sc_data.write(data_path + 'hbec.single_cell.{}.{}.h5ad'.format(tp, stim))
				
	
def separate_datasets():
	
	for ct in ['BC', 'B', 'C']:

		for tp in ['3', '6', '9', '24', '48']:

			for stim in ['alpha', 'beta']:
			
				adata = sc.read(data_path + 'hbec.single_cell.{}.{}.{}.h5ad'.format(ct,tp, stim))

				expr_df = pd.DataFrame(adata.X.toarray(), columns=adata.var.index, index=adata.obs.index)

				expr_df.to_csv(data_path + 'hbec.single_cell.{}.{}.{}.expr.csv'.format(ct,tp, stim))
				adata.obs.to_csv(data_path + 'hbec.single_cell.{}.{}.{}.obs.csv'.format(ct,tp, stim))
				adata.var.to_csv(data_path + 'hbec.single_cell.{}.{}.{}.var.csv'.format(ct,tp, stim))

def separate_combined_datasets():
	
	for tp in ['3', '6', '9', '24', '48']:

		for stim in ['alpha', 'beta']:

			adata = sc.read(data_path + 'hbec.single_cell.{}.{}.h5ad'.format(tp, stim))

			expr_df = pd.DataFrame(adata.X.toarray(), columns=adata.var.index, index=adata.obs.index)

			expr_df.to_csv(data_path + 'hbec.single_cell.{}.{}.expr.csv'.format(tp, stim))
			adata.obs.to_csv(data_path + 'hbec.single_cell.{}.{}.obs.csv'.format(tp, stim))
			adata.var.to_csv(data_path + 'hbec.single_cell.{}.{}.var.csv'.format(tp, stim))

if __name__ == '__main__':
	
	# generate_datasets()
	# generate_combined_datasets()
	# separate_datasets()
	separate_combined_datasets()