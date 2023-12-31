# correlation_estimation.py

# Authors: Min Cheol Kim

# This script generates the figure for correlation comparisons, comparing the naive, Poisson, and memento estimators

import scanpy as sc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys
sys.path.append('/home/ubuntu/Github/memento/')
import memento
import memento.auxillary.simulate as simulate
import scipy.sparse as sparse
import sklearn.datasets as sklearn_datasets
import itertools


DATA_PATH = '/home/ubuntu/Data/'
CELL_TYPE = 'CD4 T cells - ctrl'

NUM_CORR_GENES = 500 # First hundred genes display correlation
NUM_TRIALS = 20
CAPTURE_EFFICIENCIES = [0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1]
NUMBER_OF_CELLS = [50, 100, 500]

METHODS = ['ground_truth','naive', 'poisson', 'hypergeometric']



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
    
    cov_matrix = sklearn_datasets.make_spd_matrix(NUM_CORR_GENES)
    true_data = simulate.simulate_transcriptomes(n_cells=n_cells, means=z_param[0], variances=z_param[1], Nc=Nc, norm_cov=cov_matrix)
    true_data[true_data < 0] = 0
    cell_filter = true_data.sum(axis=1) > 0
    true_data = true_data[cell_filter, :]
    
    qs, captured_data = simulate.capture_sampling(true_data, q, q_sq=None)
    captured_data = sparse.csr_matrix(captured_data)
    true_data = sparse.csr_matrix(true_data)
    return true_data, captured_data


def estimate_correlations(true_data, captured_data, method, q):
    
    
    if method == 'ground_truth':
        size_factor = true_data.sum(axis=1).A1
        relative_data = true_data[:, :NUM_CORR_GENES].toarray()/size_factor.reshape(-1,1)
        mat = np.corrcoef(relative_data, rowvar=False)
        pairs = list(itertools.combinations(np.arange(NUM_CORR_GENES), 2))
        return np.array([mat[a,b] for a, b in pairs])
        
        
    elif method == 'naive':
        size_factor = captured_data.sum(axis=1).A1
        relative_data = captured_data[:, :NUM_CORR_GENES].toarray()/size_factor.reshape(-1,1)
        mat =  np.corrcoef(relative_data, rowvar=False)
        pairs = list(itertools.combinations(np.arange(NUM_CORR_GENES), 2))
        return np.array([mat[a,b] for a, b in pairs])
    
    elif method == 'hypergeometric':
        size_factor = captured_data.sum(axis=1).A1
        i1, i2 = zip(*list(itertools.combinations(np.arange(NUM_CORR_GENES), 2)))
        return memento.estimator.RNAHypergeometric(q).correlation(captured_data, size_factor, i1, i2)
    
    elif method == 'poisson':
        size_factor = captured_data.sum(axis=1).A1
        i1, i2 = zip(*list(itertools.combinations(np.arange(NUM_CORR_GENES), 2)))
        return memento.estimator.RNAPoisson().correlation(captured_data, size_factor, i1, i2)
    
    else:
        
        print('Not implemented!')
        return np.zeros(data.shape[-1])
        

if __name__ == '__main__':
    
    # Extract gene expression parameters to be used across all simulations
    x_param, z_param, Nc = get_simulation_parameters(q=0.07)
    
    # Simulation setup
    num_gene_pairs = int(NUM_CORR_GENES*(NUM_CORR_GENES-1)/2)
    num_simulations = len(CAPTURE_EFFICIENCIES) * len(NUMBER_OF_CELLS) * len(METHODS) * NUM_TRIALS
    correlation_estimates = np.zeros((num_simulations, num_gene_pairs))
    simulation_details = []
    
    simulation_idx = 0
    for q in CAPTURE_EFFICIENCIES:
        
        for num_cell in NUMBER_OF_CELLS:
            
            for trial in range(NUM_TRIALS):
                
                true_data, captured_data = simulate_data(num_cell, z_param, Nc)
                
                for method in METHODS:
                    
                    correlation_estimates[simulation_idx, :] = estimate_correlations(true_data, captured_data, method, q)
                    simulation_details.append((q, num_cell, trial+1, method))
                    simulation_idx += 1
                    
    metadata = pd.DataFrame(simulation_details, columns=['q', 'num_cell', 'trial', 'method'])
    metadata.to_csv(DATA_PATH + 'simulation/correlation/simulation_metadata.csv', index=False)
    np.savez_compressed(DATA_PATH + 'simulation/correlation/simulation_correlations', correlations=correlation_estimates)
                                                         
                    
                
                
                
                
                

                
                


    