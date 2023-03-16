import pandas as pd
import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt

data_path = '/data_volume/memento/method_comparison/lupus/'

### Read single cell and bulk data

adata = sc.read(data_path + '../../lupus/Lupus_study_adjusted_counts.h5ad')

bulk = pd.read_csv(data_path + 'lupus_bulk.csv', index_col=0)

def get_sc_ind(x):
    
    if '-' in x:
        return x.split('-')[1]
    elif '_' in x:
        return x.split('_')[0]
    else:
        return x

meta = adata.obs[['ind_cov', 'Age', 'Sex', 'SLE_status']].drop_duplicates().reset_index(drop=True)
meta['ind'] = meta['ind_cov'].apply(get_sc_ind)

sc_inds = set(meta['ind'].tolist())

bulk_inds = set(bulk.columns.str.split('_').str[1].tolist())

inds = list(sc_inds & bulk_inds)

meta = meta[meta['ind'].isin(inds)]

genes = list(set(bulk.index) & set(adata.var.index))

## Create CD14 vs CD4 comparisons

### Sample individuals

# For now, we'll stick to comparing CD14 vs CD4 cells
numinds=4
for numcells in [50, 100, 150, 200]:

	for trial in range(50):

		sampled_inds = np.random.choice(inds, numinds)

		### Create single cell data and pseudobulks

		adata.obs['ind'] = adata.obs['ind_cov'].apply(get_sc_ind)

		sampled_adata = adata[adata.obs['ind'].isin(sampled_inds) & adata.obs['cg_cov'].isin(['T4', 'cM']), genes]

		pseudobulks = []
		names = []
		adata_list = []
		for ind in sampled_inds:
			for ct in ['T4', 'cM']:
				ind_ct_adata = sampled_adata[(sampled_adata.obs['ind']==ind) & (sampled_adata.obs['cg_cov']==ct)].copy()
				sc.pp.subsample(ind_ct_adata, n_obs=numcells)
				adata_list.append(ind_ct_adata.copy())
				pseudobulks.append( ind_ct_adata.X.sum(axis=0).A1)
				names.append(('CD14' if ct == 'cM' else 'CD4') + '_' + ind )
		sc_data = sc.AnnData.concatenate(*adata_list)
		pseudobulks = np.vstack(pseudobulks)
		pseudobulks = pd.DataFrame(pseudobulks.T, columns=names, index=genes)

		pseudobulks.to_csv(data_path + 'T4_vs_cM.pseudobulk.{}.{}.csv'.format(numcells,trial))

		sc_data.write(data_path + 'T4_vs_cM.single_cell.{}.{}.h5ad'.format(numcells,trial))

		### Select bulk data

		names = []
		for ind in sampled_inds:
			for ct in ['T4', 'cM']:

				name = ('CD14' if ct == 'cM' else 'CD4') + '_' + ind
				names.append(name)
		bulk.loc[genes, names].to_csv(data_path + 'T4_vs_cM.bulk.{}.{}.csv'.format(numcells,trial))