# correlation_estimation.py

# Authors: Min Cheol Kim

# This script generates the estimates for smFISH correlation estimates

import scanpy as sc
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
import pandas as pd

import sys
import scipy.sparse as sparse


import os
import tempfile
import scvi
import numpy as np
import torch
from rich import print

DATA_PATH = '/home/ubuntu/Data/'
NUM_TRIALS = 20
NUMBER_OF_CELLS = [500, 1000, 5000, 8000]
NUMBER_OF_CELLS = [8000]
# For testing
# NUM_TRIALS = 1
# NUMBER_OF_CELLS = [500]
    
if __name__ == '__main__':
    
    dropseq_adata = sc.read_h5ad(DATA_PATH + 'smfish/filtered_dropseq.h5ad')
    num_genes = dropseq_adata.shape[1]
    dropseq_genes = dropseq_adata.var.index.tolist()
    
    smfish_estimates = np.load(DATA_PATH + 'smfish/smfish_estimates.npz')
    smfish_genes = list(smfish_estimates['corr_genes'])
    num_pairs = len(smfish_genes)
    genes_1, genes_2 = [pair[0] for pair in smfish_genes], [pair[1] for pair in smfish_genes]
    gene_idxs_1, gene_idxs_2 = [dropseq_genes.index(g) for g in genes_1], [dropseq_genes.index(g) for g in genes_2]
    all_genes = list(set(genes_1) & set(genes_2))
        
    for num_cell in NUMBER_OF_CELLS:

        for trial in range(NUM_TRIALS if num_cell < 5000 else 1):

            print('num cell', num_cell, 'trial', trial)
            adata = sc.read_h5ad(DATA_PATH + f'smfish/correlation/{num_cell}_{trial}.h5ad')
            adata.var.index = dropseq_adata.var.index.tolist()
            adata.layers['counts'] = adata.X
            scvi.model.SCVI.setup_anndata(adata, layer="counts")
            model = scvi.model.SCVI(adata, n_layers=2, n_latent=30, gene_likelihood="nb")
            model.train()

            corr_matrix = model.get_feature_correlation_matrix(correlation_type='pearson')
            correlations = corr_matrix.loc[all_genes, all_genes]
            correlations.to_csv(DATA_PATH + f'smfish/correlation/{num_cell}_{trial}_scvi_corr.csv')                             
                    
                
                
                
                
                

                
                


    