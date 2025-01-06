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

NUM_TRIALS = 500
METHODS = ['naive', 'hypergeometric']
NUMBER_OF_CELLS = [10, 20, 30, 50,100, 5000, 8000]
NUM_SAMPLES = (len(NUMBER_OF_CELLS)-1) * len(METHODS) * NUM_TRIALS + len(METHODS)



def sample_dropseq_data(adata, num_cells):
    """ Generates sampled data. """
    
    if num_cells >= 8000:
        
        return adata.copy()
    
    random_idxes = np.random.choice(adata.shape[0], num_cells, replace=False)
    
    return adata[random_idxes].copy()


def estimate_mean(data, method):
    
    size_factor = data.sum(axis=1).A1
    
    if method == 'naive':
        
        return (data/size_factor.reshape(-1,1)).mean(axis=0).A1
    
    elif method == 'hypergeometric':
        
        est = memento.estimator.RNAHypergeometric(0.01485030176341905)
        sf = est.estimate_size_factor(data, shrinkage=0.5, filter_mean_thresh=0.07, trim_percent=0.1)
        return est.mean(data, sf)
    else:
        
        print('Not implemented!')
        return np.zeros(data.shape[-1])
        

if __name__ == '__main__':
    
    dropseq_adata = sc.read_h5ad(DATA_PATH + 'smfish/dropseq.h5ad')
    
    ncounts = dropseq_adata.X.sum(axis=1).A1
    dropseq_adata = dropseq_adata[(ncounts > 700) & (ncounts < 8000)]
    # np.random.shuffle(dropseq_adata.X.data)
    num_genes = dropseq_adata.shape[1]
    
    # Analysis setup
    mean_estimates = np.zeros((NUM_SAMPLES, num_genes), dtype=np.float64)
    sample_details = []
    
    sample_idx = 0
        
    for num_cell in NUMBER_OF_CELLS:

        for trial in range(NUM_TRIALS if num_cell < 8000 else 1):

            # Iterate through the methods and calculate mean
            sampled = sample_dropseq_data(dropseq_adata, num_cell)
            for method in METHODS:
                
                mean_estimates[sample_idx, :] = estimate_mean(sampled.X, method)
                sample_details.append((num_cell, trial+1, method))
                sample_idx += 1
                    
    metadata = pd.DataFrame(sample_details, columns=['num_cell', 'trial', 'method'])
    metadata.to_csv(DATA_PATH + 'smfish/mean/sample_metadata.csv', index=False)
    np.savez_compressed(DATA_PATH + 'smfish/mean/sample_means', means=mean_estimates)