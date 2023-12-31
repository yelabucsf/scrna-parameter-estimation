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
CELL_TYPE = 'CD4 T cells - ctrl'

NUM_TRIALS = 20
CAPTURE_EFFICIENCIES = [0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1]
NUMBER_OF_CELLS = [50, 100, 500]

def concordance(x, y, log=True):
    
    if log:
        a = np.log(x)
        b = np.log(y)
    else:
        a = x
        b = y
    cond = np.isfinite(a) & np.isfinite(b)
    a = a[cond]
    b = b[cond]
    cmat = np.cov(a, b)
    return 2*cmat[0,1]/(cmat[0,0] + cmat[1,1] + (a.mean()-b.mean())**2)


def get_simulation_parameters(q=0.07):
    """ Extracts simulation parameters. """
    
    adata = sc.read(DATA_PATH + 'interferon_filtered.h5ad')
    adata = adata[adata.obs.cell_type == CELL_TYPE]
    data = adata.X.copy()
    relative_data = data.toarray()/data.sum(axis=1)
    
    x_param, z_param, Nc, good_idx = simulate.extract_parameters(adata.X, q=q, min_mean=0.01)
    
    return x_param, z_param, Nc


def simulate_data(n_cells, z_param, Nc):
    """ Generates simulated data. """
    
    true_data = simulate.simulate_transcriptomes(n_cells=n_cells, means=z_param[0], variances=z_param[1], Nc=Nc, norm_cov='uncorrelated')
    true_data[true_data < 0] = 0
    
    qs, captured_data = simulate.capture_sampling(true_data, q, q_sq=None)
    true_data = sparse.csr_matrix(true_data)
    captured_data = sparse.csr_matrix(captured_data)
    
    return true_data, captured_data
        

if __name__ == '__main__':
    
    # Extract gene expression parameters to be used across all simulations
    x_param, z_param, Nc = get_simulation_parameters(q=0.07)
    true_mean = x_param[0]
    
    simulation_idx = 1
    for q in CAPTURE_EFFICIENCIES:
        
        for num_cell in NUMBER_OF_CELLS:
            
            for trial in range(NUM_TRIALS):
                
                true_data, captured_data = simulate_data(num_cell, z_param, Nc)
                size_factor = captured_data.sum(axis=1).A1

                sc.AnnData(true_data).write(DATA_PATH + f'simulation/variance/{num_cell}_{q}_{trial}_ground_truth.h5ad')
                sc.AnnData(captured_data).write(DATA_PATH + f'simulation/variance/{num_cell}_{q}_{trial}.h5ad')

                
                    
                
                
                
                
                

                
                


    