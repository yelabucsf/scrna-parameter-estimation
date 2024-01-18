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
MIN_MEAN_THRESH = 0.01

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
    
    dropseq_adata.write(DATA_PATH + 'smfish/filtered_dropseq.h5ad')

    # Calculate stuff for smFISH
    smfish = pd.read_csv(DATA_PATH + 'smfish/fishSubset.txt', index_col=0, sep=' ')
    filtered_fish = smfish.query('GAPDH > 0')
    overlap_genes = list(set(dropseq_genes) & set(smfish.columns))

    mean_genes = overlap_genes
    var_genes = [i for i in overlap_genes if i != 'GAPDH']
    corr_genes = [(a,b) for a,b in itertools.combinations(overlap_genes, 2) if 'GAPDH' not in [a,b]]

    smfish_means = np.zeros(len(mean_genes))
    smfish_variances = np.zeros(len(var_genes))
    smfish_correlations = np.zeros(len(corr_genes))

    for idx, gene in enumerate(mean_genes):
        if gene == 'GAPDH':
            smfish_means[idx] = 1.0
        df = filtered_fish[['GAPDH', gene]].dropna()
        norm = df[gene].values/df['GAPDH'].values
        smfish_means[idx] = norm.mean()

    for idx, gene in enumerate(var_genes):

        df = filtered_fish[['GAPDH', gene]].dropna()
        norm = df[gene].values/df['GAPDH'].values
        smfish_variances[idx] = norm.var()

    for idx, pair in enumerate(corr_genes):

        gene1, gene2 = pair        
        df = filtered_fish[[gene1, gene2, 'GAPDH']].dropna()

        if df.shape[0] < 2:
            smfish_correlations[idx] = np.nan
            continue
        norm1 = (df[gene1]/df['GAPDH']).values
        norm2 = (df[gene2]/df['GAPDH']).values
        smfish_correlations[idx] = stats.pearsonr(norm1, norm2)[0]


    np.savez_compressed(
        DATA_PATH + 'smfish/smfish_estimates',
        mean_genes=np.array(mean_genes),
        var_genes=np.array(var_genes),
        corr_genes = np.array(corr_genes),
        mean=smfish_means,
        variance=smfish_variances,
        correlation=smfish_correlations)