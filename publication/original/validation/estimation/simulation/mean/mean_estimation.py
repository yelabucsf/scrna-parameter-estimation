# simulate_mean_estimation.py

# Authors: Min Cheol Kim

# This script generates the figure for mean comparisons, comparing the naive mean estimator and the memento estimator. 

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
METHODS = ['naive', 'pb', 'hypergeometric']
CAPTURE_EFFICIENCIES = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1]
NUMBER_OF_CELLS = [10, 20, 30, 40, 50, 100, 200, 500]
NUMBER_OF_CELLS = [10, 500]


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
    captured_data = sparse.csr_matrix(captured_data)
    
    return true_data, captured_data


def estimate_means(data, size_factor, method):
    
    
    if method == 'naive':
        return (data/size_factor.reshape(-1,1)).mean(axis=0).A1

    elif method == 'pb':
        return data.sum(axis=0).A1/data.sum()
    
    elif method == 'hypergeometric':
        estimator = memento.estimator.RNAHypergeometric(0.07)
        sf = estimator.estimate_size_factor(data)
        return estimator.mean(data, sf)
    else:
        
        print('Not implemented!')
        return np.zeros(data.shape[-1])
        

if __name__ == '__main__':
    
    # Extract gene expression parameters to be used across all simulations
    x_param, z_param, Nc = get_simulation_parameters(q=0.07)
    true_mean = x_param[0]
    
    # Simulation setup
    num_genes = true_mean.shape[0]
    num_simulations = len(CAPTURE_EFFICIENCIES) * len(NUMBER_OF_CELLS) * len(METHODS) * NUM_TRIALS
    mean_estimates = np.zeros((num_simulations+1, num_genes))
    simulation_details = []
    
    # First "simulation" is the true data
    mean_estimates[0, :] = true_mean
    simulation_details.append((1, np.inf, 0, 'ground_truth'))
    
    simulation_idx = 1
    for q in CAPTURE_EFFICIENCIES:
        
        for num_cell in NUMBER_OF_CELLS:
            
            for trial in range(NUM_TRIALS):
                
                true_data, captured_data = simulate_data(num_cell, z_param, Nc)
                size_factor = captured_data.sum(axis=1).A1
                
                for method in METHODS:
                    
                    mean_estimates[simulation_idx, :] = estimate_means(captured_data, size_factor, method)
                    simulation_details.append((q, num_cell, trial+1, method))
                    simulation_idx += 1
    metadata = pd.DataFrame(simulation_details, columns=['q', 'num_cell', 'trial', 'method'])
    metadata.to_csv(DATA_PATH + 'simulation/mean/simulation_metadata.csv', index=False)
    np.savez_compressed(DATA_PATH + 'simulation/mean/simulation_means', means=mean_estimates)
                                                         
                    
                
                
                
                
                

                
                


    