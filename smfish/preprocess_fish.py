import pandas as pd
import scanpy as sc
import seaborn as sns
import scanpy as sc
import scipy.sparse as sparse
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import scipy.optimize as opt
import itertools

DATA_PATH = '/home/ubuntu/Data/'
MIN_MEAN_THRESH = 0.005

if __name__ == '__main__':
    
#     LEGACY CODE, ONLY FOR REVERSE ENGINEERING N_COUNTS
#     data_path = '/data_volume/memento/saver/'

#     partial = pd.read_csv('/data_volume/memento/saver/melanoma_dropseq.csv', index_col=0, sep=',').T
#     dropseq = pd.read_csv('GSE99330_dropseqRPM.txt', sep=' ').T.loc[partial.index]/1e6

#     # Reverse calculate cell sizes
#     high_expr = partial.mean(axis=0).sort_values().tail(20).index.tolist()
#     dropseq_sf = (partial[high_expr]/dropseq[high_expr]).mean(axis=1)
#     dropseq = dropseq*dropseq_sf.values.reshape(-1,1)

#     # Create sparse matrix and save
#     X = sparse.csr_matrix(dropseq.values)
#     dropseq_adata = sc.AnnData(X=X, obs=pd.DataFrame(index=dropseq.index), var=pd.DataFrame(index=dropseq.columns))

#     dropseq_adata.write(data_path + 'full_dropseq.h5ad')

    # Calculate stuff for Dropseq data
    dropseq_adata = sc.read_h5ad(DATA_PATH + 'smfish/full_dropseq.h5ad')
    dropseq_adata = dropseq_adata[dropseq_adata.X.sum(axis=1).A1 > 0].copy()

    dropseq_adata.obs['n_counts'] = dropseq_adata.X.sum(axis=1).A1
    dropseq_adata.obs['n_genes'] = (dropseq_adata.X > 0).sum(axis=1).A1

    z_means = dropseq_adata.X.mean(axis=0).A1    
    dropseq_genes = dropseq_adata.var.index[z_means > MIN_MEAN_THRESH].tolist()
    
    dropseq_adata.obs.index = [f'cell{i}' for i in range(dropseq_adata.shape[0])]
    dropseq_adata.var.index = [f'gene{i}' for i in range(dropseq_adata.shape[1])]
    dropseq_adata.write(DATA_PATH + 'smfish/filtered_dropseq.h5ad')

    # Calculate stuff for smFISH
    smfish = pd.read_csv(DATA_PATH + 'smfish/fishSubset.txt', index_col=0, sep=' ')
    filtered_fish = smfish.query('GAPDH > 0')
    overlap_genes = list(set(dropseq_genes) & set(smfish.columns))
    norm_smfish = (filtered_fish[overlap_genes]/(filtered_fish['GAPDH']).values.reshape(-1,1)).values # np array
    
    smfish_mean = norm_smfish.mean(axis=0)
    smfish_variance = norm_smfish.var(axis=0)
    smfish_correlation = np.corrcoef(norm_smfish, rowvar=False)
    
    np.savez_compressed(
        DATA_PATH + 'smfish/smfish_estimates',
        genes=np.array(overlap_genes),
        mean=smfish_mean,
        variance=smfish_variance,
        correlation=smfish_correlation)
