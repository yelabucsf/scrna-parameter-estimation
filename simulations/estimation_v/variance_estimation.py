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
CAPTURE_EFFICIENCIES = [0.05, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1]
NUMBER_OF_CELLS = [50, 100, 500]
METHODS = ['ground_truth','naive', 'poisson', 'hypergeometric', 'basics']


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
    
    if method == 'ground_truth':
        
        # Calculate ground truth (pre-sampling) parameters
        true_data = sc.read(DATA_PATH + f'simulation/variance/{num_cell}_{q}_{trial}_ground_truth.h5ad').X
        true_size_factor = true_data.sum(axis=1).A1
        true_mean = (true_data/true_size_factor.reshape(-1,1)).mean(axis=0)
        true_variance = (true_data.toarray()/true_size_factor.reshape(-1,1)).var(axis=0)
        
        return true_variance
    
    elif method == 'naive':
        
        return (data.toarray()/size_factor.reshape(-1,1)).var(axis=0)

    elif method == 'poisson':
        
        return memento.estimator.RNAPoisson().variance(data, size_factor)
    
    elif method == 'hypergeometric':
        
        return memento.estimator.RNAHypergeometric(q).variance(data, size_factor)
    
    elif method == 'basics': # just load the values already calculated from BASICS R script
        
        basics_parameters = pd.read_csv(DATA_PATH + f'simulation/variance/{num_cell}_{q}_{trial}_parameters.csv', index_col=0)
        scale_factor = (basics_parameters['mu']/x_param[0]).mean()
        basics_parameters['scaled_variance'] = basics_parameters['variance']/scale_factor**2

        return basics_parameters['scaled_variance'].values
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
    variance_estimates = np.zeros((num_simulations, num_genes), dtype=np.float64)
    simulation_details = []
    
    simulation_idx = 0
    for q in CAPTURE_EFFICIENCIES:
        
        for num_cell in NUMBER_OF_CELLS:
            
            for trial in range(NUM_TRIALS):
                
                # Iterate through the methods and calculate variance
                for method in METHODS:
                    
                    variance_estimates[simulation_idx, :] = estimate_variance(method, q, num_cell, trial, x_param)
                    simulation_details.append((q, num_cell, trial+1, method))
                    simulation_idx += 1
                    
    metadata = pd.DataFrame(simulation_details, columns=['q', 'num_cell', 'trial', 'method'])
    metadata.to_csv(DATA_PATH + 'simulation/variance/simulation_metadata.csv', index=False)
    np.savez_compressed(DATA_PATH + 'simulation/variance/simulation_variances', means=variance_estimates)
                                                         
                    
                
                
                
                
                

                
                


    