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

NUM_TRIALS = 1
METHODS = ['naive', 'hypergeometric']
NUMBER_OF_CELLS = [50, 100, 500, 1000, 5000]


def sample_dropseq_data(adata, num_cells):
    """ Generates sampled data. """
    
    random_idxes = np.random.choice(adata.shape[0], num_cells, replace=False)
    
    return adata[random_idxes].copy()


def estimate_mean(data, method):
    
    size_factor = data.sum(axis=1).A1
    
    if method == 'naive':
        return (data/size_factor.reshape(-1,1)).mean(axis=0).A1
    
    elif method == 'hypergeometric':
        
        return memento.estimator.RNAHypergeometric(0.07).mean(data, size_factor)
    else:
        
        print('Not implemented!')
        return np.zeros(data.shape[-1])
        

if __name__ == '__main__':
    
    dropseq_adata = sc.read_h5ad(DATA_PATH + 'smfish/filtered_dropseq.h5ad')
    num_genes = dropseq_adata.shape[1]
    
    # Analysis setup
    num_samples = len(NUMBER_OF_CELLS) * len(METHODS) * NUM_TRIALS
    mean_estimates = np.zeros((num_samples, num_genes), dtype=np.float64)
    sample_details = []
    
    sample_idx = 0
        
    for num_cell in NUMBER_OF_CELLS:

        for trial in range(NUM_TRIALS):

            # Iterate through the methods and calculate mean
            sampled = sample_dropseq_data(dropseq_adata, num_cell)
            for method in METHODS:

                mean_estimates[sample_idx, :] = estimate_mean(sampled.X, method)
                sample_details.append((num_cell, trial+1, method))
                sample_idx += 1
                    
    metadata = pd.DataFrame(sample_details, columns=['num_cell', 'trial', 'method'])
    metadata.to_csv(DATA_PATH + 'smfish/mean/subsample_metadata.csv', index=False)
    np.savez_compressed(DATA_PATH + 'smfish/mean/subsample_means', means=mean_estimates)
                                                         
                    
                
                
                
                
                

                
                


    