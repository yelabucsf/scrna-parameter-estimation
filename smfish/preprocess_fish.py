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

DATA_PATH = '/home/ubuntu/Data/saver/'
MIN_MEAN_THRESH = 0.005

if __name__ == '__main__':
    
    
    # Calculate stuff for Dropseq data
    dropseq_adata = sc.read(DATA_PATH + 'full_dropseq.h5ad')
    dropseq_adata = dropseq_adata[dropseq_adata.X.sum(axis=1).A1 > 0].copy()

    dropseq_adata.obs['n_counts'] = dropseq_adata.X.sum(axis=1).A1
    dropseq_adata.obs['n_genes'] = (dropseq_adata.X > 0).sum(axis=1).A1

    z_means = dropseq_adata.X.mean(axis=0).A1    
    dropseq_genes = dropseq_adata.var.index[z_means > MIN_MEAN_THRESH].tolist()

    # Calculate stuff for smFISH
    smfish = pd.read_csv(DATA_PATH + 'fishSubset.txt', index_col=0, sep=' ')
    overlap_genes = list(set(dropseq_genes) & set(smfish.columns))
    
    