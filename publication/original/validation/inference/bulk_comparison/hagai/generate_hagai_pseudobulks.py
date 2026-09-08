import pandas as pd
import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
import sys

import memento

data_path = '/data_volume/bulkrna/hagai/'

files = [
    'Hagai2018_mouse-lps',
    'Hagai2018_mouse-pic',
    'Hagai2018_pig-lps',
    'Hagai2018_rabbit-lps',
    'Hagai2018_rat-lps',
    'Hagai2018_rat-pic',
]


def generate_pseudobulks():
    
    for fname in files:

        print('working on', fname)
        
        adata = sc.read(data_path + f'sc_rnaseq/h5Seurat/{fname}.h5ad')
                        
        inds = adata.obs['replicate'].drop_duplicates().tolist()
        stims = adata.obs['label'].drop_duplicates().tolist()

        pseudobulks = []
        names = []
        for ind in inds:
            for stim in stims:
                ind_stim_adata = adata[(adata.obs['replicate']==ind) & (adata.obs['label']==stim)].copy()
                pseudobulks.append( ind_stim_adata.X.sum(axis=0).A1)
                names.append(stim + '_' + ind )

        pseudobulks = np.vstack(pseudobulks)
        pseudobulks = pd.DataFrame(pseudobulks.T, columns=names, index=adata.var.index.tolist())
        
        out_name = fname
        pseudobulks.to_csv(data_path + 'sc_rnaseq/pseudobulks/' + out_name + '.csv')
        
def generate_sampled_data(num_cells):
    
    for fname in files:

        print('working on', fname)
        
        adata = sc.read(data_path + f'sc_rnaseq/h5Seurat/{fname}.h5ad')
                        
        inds = adata.obs['replicate'].drop_duplicates().tolist()
        stims = adata.obs['label'].drop_duplicates().tolist()

        pseudobulks = []
        names = []
        adata_list = []
        for ind in inds:
            for stim in stims:
                ind_stim_adata = adata[(adata.obs['replicate']==ind) & (adata.obs['label']==stim)].copy()
                
                sc.pp.subsample(ind_stim_adata, n_obs=num_cells)
                pseudobulks.append( ind_stim_adata.X.sum(axis=0).A1)
                names.append(stim + '_' + ind )
                adata_list.append(ind_stim_adata)
        
        sc_data = sc.AnnData.concatenate(*adata_list)
        sc_data.write(data_path + f'sc_rnaseq/h5Seurat/{fname}_{num_cells}.h5ad')

        pseudobulks = np.vstack(pseudobulks)
        pseudobulks = pd.DataFrame(pseudobulks.T, columns=names, index=adata.var.index.tolist())
        
        pseudobulks.to_csv(data_path + f'sc_rnaseq/pseudobulks/{fname}_{num_cells}.csv')

        
if __name__ == '__main__':

    # generate_pseudobulks()
    
    generate_sampled_data(num_cells=100)