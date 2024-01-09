# simulate_variance_datasets.py

# Authors: Min Cheol Kim

# This script generates datasets for variance estimation comparisons

import scanpy as sc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys
sys.path.append('/home/ubuntu/Github/memento/')
import memento
import memento.auxillary.simulate as simulate
import scipy.sparse as sparse

DATA_PATH = '/home/ubuntu/Data/'

NUM_TRIALS = 20
NUMBER_OF_CELLS = [50, 100, 500, 1000, 5000]

def sample_dropseq_data(adata, num_cells):
    """ Generates sampled data. """
    
    random_idxes = np.random.choice(adata.shape[0], num_cells, replace=False)
    
    sampled_adata =  adata[random_idxes].copy()
        
    sampled_adata.obs.index = [f'cell{i}' for i in range(sampled_adata.shape[0])]
    sampled_adata.var.index = [f'gene{i}' for i in range(sampled_adata.shape[1])]
    
    return sampled_adata

if __name__ == '__main__':
    
    dropseq_adata = sc.read_h5ad(DATA_PATH + 'smfish/filtered_dropseq.h5ad')
        
    for num_cell in NUMBER_OF_CELLS:

        for trial in range(NUM_TRIALS):

            sampled = sample_dropseq_data(dropseq_adata, num_cell)
            
            sampled.write(DATA_PATH + f'smfish/variance/{num_cell}_{trial}.h5ad')
            
    dropseq_adata.write(DATA_PATH + f'smfish/variance/8000_0.h5ad')

                
                    
                
                
                
                
                

                
                


    