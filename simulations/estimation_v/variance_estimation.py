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
CAPTURE_EFFICIENCIES = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1]
NUMBER_OF_CELLS = [50, 100, 200, 300, 500]
METHODS = ['naive', 'poisson', 'basics', 'hypergeometric']


def get_simulation_parameters(q=0.07):
    """ Extracts simulation parameters. """
    
    adata = sc.read(DATA_PATH + 'interferon_filtered.h5ad')
    adata = adata[adata.obs.cell_type == CELL_TYPE]
    data = adata.X.copy()
    relative_data = data.toarray()/data.sum(axis=1)
    
    x_param, z_param, Nc, good_idx = simulate.extract_parameters(adata.X, q=q, min_mean=0.01)
    
    return x_param, z_param, Nc


def estimate_variance(method, q, nc, trial, x_param): # x_param used for scaling for basics
    
    captured_data = sc.read(DATA_PATH + f'simulation/variance/{num_cell}_{q}_{trial}.h5ad')
    data = captured_data.X
    size_factor = data.sum(axis=1).A1    
    
    if method == 'naive':
        
        return (data.toarray()/size_factor.reshape(-1,1)).var(axis=0)

    elif method == 'poisson':
        
        return memento.estimator.RNAPoisson().variance(data, size_factor)
    
    elif method == 'hypergeometric':
        
        return memento.estimator.RNAHypergeometric(0.07).variance(data, size_factor)
    
    elif method == 'basics':
        
        basics_parameters = pd.read_csv(DATA_PATH + f'simulation/variance/{num_cell}_{q}_{trial}_parameters.csv', index_col=0)
        scale_factor = (basics_parameters['mu']/x_param[0]).mean()
        basics_parameters['scaled_variance'] = basics_parameters['variance']/scale_factor**2
    else:
        
        print('Not implemented!')
        return np.zeros(data.shape[-1])
        

if __name__ == '__main__':
    
    # Extract gene expression parameters to be used across all simulations
    x_param, z_param, Nc = get_simulation_parameters(q=0.07)
    true_variance = x_param[1]
    
    # Simulation setup
    num_genes = true_variance.shape[0]
    num_simulations = len(CAPTURE_EFFICIENCIES) * len(NUMBER_OF_CELLS) * len(METHODS) * NUM_TRIALS
    variance_estimates = np.zeros((num_simulations+1, num_genes))
    simulation_details = []
    
    # First "simulation" is the true data
    variance_estimates[0, :] = true_variance
    simulation_details.append((1, np.inf, 0, 'ground_truth'))
    
    simulation_idx = 1
    for q in CAPTURE_EFFICIENCIES:
        
        for num_cell in NUMBER_OF_CELLS:
            
            for trial in range(NUM_TRIALS):
                
                for method in METHODS:
                    
                    variance_estimates[simulation_idx, :] = estimate_variance(method, q, num_cell, trial, x_param)
                    simulation_details.append((q, num_cell, trial+1, method))
                    simulation_idx += 1
    metadata = pd.DataFrame(simulation_details, columns=['q', 'num_cell', 'trial', 'method'])
    metadata.to_csv(DATA_PATH + 'simulation/variance/simulation_metadata.csv', index=False)
    np.savez_compressed(DATA_PATH + 'simulation/variance/simulation_variances', means=variance_estimates)
                                                         
                    
                
                
                
                
                

                
                


    