import pandas as pd
import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/ssm-user/Github/scrna-parameter-estimation/dist/memento-0.1.0-py3.10.egg')
import memento

data_path = '/data_volume/memento/hagai/'

files = [
    'Hagai2018_mouse-lps',
    'Hagai2018_mouse-pic',
    'Hagai2018_pig-lps',
    'Hagai2018_rabbit-lps',
    'Hagai2018_rat-lps',
    'Hagai2018_rat-pic',
]


def generate_pseudobulks(shuffled=False):
    
    for fname in files:

        print('working on', fname)
        
        adata = sc.read(data_path + f'sc_rnaseq/h5Seurat/{fname}.h5ad')
        
        if shuffled:
            adata.obs['label'] = adata.obs['label'].sample(frac=1).values
                        
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
        
        out_name = fname + '_shuffled' if shuffled else fname
        pseudobulks.to_csv(data_path + 'sc_rnaseq/pseudobulks/' + out_name + '.csv')

        
if __name__ == '__main__':

    generate_pseudobulks(shuffled=False)
    generate_pseudobulks(shuffled=True)