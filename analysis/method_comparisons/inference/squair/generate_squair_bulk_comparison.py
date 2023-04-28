import pandas as pd
import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/ssm-user/Github/scrna-parameter-estimation/dist/memento-0.1.0-py3.10.egg')
import memento

data_path = '/data_volume/memento/method_comparison/squair/'

files = [
    'Hagai2018_mouse-lps',
    'Hagai2018_mouse-pic',
    'Hagai2018_pig-lps',
    'Hagai2018_rabbit-lps',
    'Hagai2018_rat-lps',
    'Hagai2018_rat-pic',
]


def scaled_mean_se2(data, sf, q):

    augmented_data = np.append(data, np.ones((1,data.shape[1])), axis=0)

    sf = np.append(sf, sf.mean())
    q = q.mean()
    X = augmented_data/sf.reshape(-1,1)

    naive_v = X.var(axis=0)
    naive_m = X.mean(axis=0)
    v = naive_v-(1-q)*(augmented_data/(sf**2-sf*(1-q)).reshape(-1,1)).mean(axis=0)
    variance_contributions = ((1-q)/sf).reshape(-1,1)*naive_m.reshape(1,-1) + v.reshape(1,-1)
    m = np.average( X, weights=1/variance_contributions, axis=0)
    m[~np.isfinite(m)] = naive_m[~np.isfinite(m)]
    m[m<0] = 0
    # return np.log(naive_m), v/data.shape[0]
    total = data.sum()
    # return m*total, v*total**2
    return m*total, (v/data.shape[0])*total**2


def generate_pseudobulks():
	
    for fname in files:
        
        print('working on', fname)

        adata = sc.read(data_path + 'sc_rnaseq/h5Seurat/' + fname + '.h5ad')
        adata.obs['q'] = 0.07
        memento.setup_memento(adata, q_column='q', trim_percent=0.05)
        
        inds = adata.obs['replicate'].drop_duplicates().tolist()
        stims = adata.obs['label'].drop_duplicates().tolist()

        pseudobulks = []
        memento_pseudobulks = []
        names = []
        adata_list = []
        for ind in inds:
            for stim in stims:
                ind_stim_adata = adata[(adata.obs['replicate']==ind) & (adata.obs['label']==stim)].copy()
                sc.pp.subsample(ind_stim_adata, n_obs=100)

                data = ind_stim_adata.X.toarray()
                sf = ind_stim_adata.obs['memento_size_factor'].values
                q = ind_stim_adata.obs['q'].values
                s, se2 = scaled_mean_se2(data, sf, q)
                
                memento_pseudobulks.append(s)
                pseudobulks.append( ind_stim_adata.X.sum(axis=0).A1)
                
                names.append(stim + '_' + ind )
                adata_list.append(ind_stim_adata.copy())
                
        memento_pseudobulks = np.vstack(memento_pseudobulks)
        memento_pseudobulks = pd.DataFrame(memento_pseudobulks.T, columns=names, index=adata.var.index.tolist())
        
        pseudobulks = np.vstack(pseudobulks)
        pseudobulks = pd.DataFrame(pseudobulks.T, columns=names, index=adata.var.index.tolist())
        
        sc_adata = sc.AnnData.concatenate(*adata_list)
        
        sc_adata.write(data_path + 'sc_rnaseq/h5Seurat/' + fname + '.h5ad')
        pseudobulks.to_csv(data_path + 'sc_rnaseq/pseudobulks/' + fname + '.csv')
        memento_pseudobulks.to_csv(data_path + 'sc_rnaseq/pseudobulks/memento_' + fname + '.csv')
	
if __name__ == '__main__':

	generate_pseudobulks()