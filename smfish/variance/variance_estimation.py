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
NUM_TRIALS = 20
NUMBER_OF_CELLS = [5000]
# NUMBER_OF_CELLS = [8000]

METHODS = ['naive','hypergeometric', 'poisson']


def estimate_variance(method, nc, trial): # x_param used for scaling for basics
    
    captured_data = sc.read_h5ad(DATA_PATH + f'smfish/variance/{num_cell}_{trial}.h5ad')
    data = captured_data.X
    size_factor = data.sum(axis=1).A1
    
    if method == 'naive':
        
        return (data.toarray()/size_factor.reshape(-1,1)).var(axis=0)

    elif method == 'poisson':
        
        return memento.estimator.RNAPoisson().variance(data, size_factor)
    
    elif method == 'hypergeometric':
        
        return memento.estimator.RNAHypergeometric(0.1).variance(data, size_factor)
    
    elif method == 'basics': # just load the values already calculated from BASICS R script
        
        basics_parameters = pd.read_csv(DATA_PATH + f'smfish/variance/{num_cell}_{q}_{trial}_parameters.csv', index_col=0)
        scale_factor = (basics_parameters['mu']/x_param[0]).mean()
        basics_parameters['scaled_variance'] = basics_parameters['variance']/scale_factor**2

        return basics_parameters['scaled_variance'].values
    else:
        
        print('Not implemented!')
        return np.zeros(data.shape[-1])

    
def estimate_mean(method, nc, trial): # x_param used for scaling for basics
    
    captured_data = sc.read_h5ad(DATA_PATH + f'smfish/variance/{num_cell}_{trial}.h5ad')
    data = captured_data.X
    size_factor = data.sum(axis=1).A1

    if method == 'naive':

        return (data.toarray()/size_factor.reshape(-1,1)).mean(axis=0)

    elif method == 'poisson':

        return memento.estimator.RNAPoisson().mean(data, size_factor)

    elif method == 'hypergeometric':

        return memento.estimator.RNAHypergeometric(0.1).mean(data, size_factor)

    elif method == 'basics': # just load the values already calculated from BASICS R script

        basics_parameters = pd.read_csv(DATA_PATH + f'smfish/variance/{num_cell}_{q}_{trial}_parameters.csv', index_col=0)
        scale_factor = (basics_parameters['mu']/x_param[0]).mean()
        basics_parameters['scaled_mean'] = basics_parameters['mu']/scale_factor

        return basics_parameters['scaled_mean'].values
    else:

        print('Not implemented!')
        return np.zeros(data.shape[-1])

    
if __name__ == '__main__':
    
    dropseq_adata = sc.read_h5ad(DATA_PATH + 'smfish/filtered_dropseq.h5ad')
    num_genes = dropseq_adata.shape[1]
    
    # Analysis setup
    num_samples = len(NUMBER_OF_CELLS) * len(METHODS) * NUM_TRIALS
    variance_estimates = np.zeros((num_samples, num_genes), dtype=np.float64)
    mean_estimates = np.zeros((num_samples, num_genes), dtype=np.float64)
    sample_details = []
    
    sample_idx = 0
        
    for num_cell in NUMBER_OF_CELLS:

        for trial in range(NUM_TRIALS):

            # Iterate through the methods and calculate variance
            for method in METHODS:

                variance_estimates[sample_idx, :] = estimate_variance(method, num_cell, trial)
                mean_estimates[sample_idx, :] = estimate_mean(method, num_cell, trial)
                sample_details.append((num_cell, trial+1, method))
                sample_idx += 1
                    
    metadata = pd.DataFrame(sample_details, columns=['num_cell', 'trial', 'method'])
    metadata.to_csv(DATA_PATH + 'smfish/variance/sample_metadata.csv', index=False)
    np.savez_compressed(DATA_PATH + 'smfish/variance/sample_variances', variances=variance_estimates)
    np.savez_compressed(DATA_PATH + 'smfish/variance/sample_means', means=mean_estimates)

                                                         
                    
                
                
                
                
                

                
                


    