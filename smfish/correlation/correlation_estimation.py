# correlation_estimation.py

# Authors: Min Cheol Kim

# This script generates the estimates for smFISH correlation estimates

import scanpy as sc
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
import pandas as pd

import sys
sys.path.append('/home/ubuntu/Github/memento/')
import memento
import memento.auxillary.simulate as simulate
import scipy.sparse as sparse


DATA_PATH = '/home/ubuntu/Data/'
NUM_TRIALS = 100
# NUMBER_OF_CELLS = [5000]
NUMBER_OF_CELLS = [500, 1000, 5000, 8000]

METHODS = ['naive','hypergeometric', 'poisson'] # add saver


def estimate_correlations(method, nc, trial): # x_param used for scaling for basics
    
    captured_data = sc.read_h5ad(DATA_PATH + f'smfish/variance/{num_cell}_{trial}.h5ad')
    data = captured_data.X
    size_factor = data.sum(axis=1).A1
    size_factor = np.ones(captured_data.shape[0]) # no normalization
    
    if method == 'naive':
        
        mat = data.toarray()/size_factor.reshape(-1,1)
        correlations = np.array([stats.pearsonr( mat[:, i1], mat[:, i2] )[0] for i1,i2 in zip(gene_idxs_1, gene_idxs_2)])
        return correlations

    elif method == 'poisson':
        
        return memento.estimator.RNAPoisson().correlation(data, size_factor, gene_idxs_1, gene_idxs_2)
    
    elif method == 'hypergeometric':
        
        return memento.estimator.RNAHypergeometric(0.01485030176341905).correlation(data, size_factor, gene_idxs_1, gene_idxs_2)
    
    elif method == 'saver': # just load the values already calculated from BASICS R script
        
        basics_parameters = pd.read_csv(DATA_PATH + f'smfish/variance/{num_cell}_{trial}_parameters.csv', index_col=0)
        naive_means = (data.toarray()/size_factor.reshape(-1,1)).mean(axis=0)
        scale_factor = basics_parameters['mu'].values/naive_means
        scale_factor = scale_factor[np.isfinite(scale_factor)].mean()
        basics_parameters['scaled_variance'] = basics_parameters['variance']/scale_factor**2
        
        return basics_parameters['scaled_variance'].values
    else:
        
        print('Not implemented!')
        return np.zeros(data.shape[-1])

    
if __name__ == '__main__':
    
    dropseq_adata = sc.read_h5ad(DATA_PATH + 'smfish/filtered_dropseq.h5ad')
    num_genes = dropseq_adata.shape[1]
    dropseq_genes = dropseq_adata.var.index.tolist()
    
    smfish_estimates = np.load(DATA_PATH + 'smfish/smfish_estimates.npz')
    smfish_genes = list(smfish_estimates['corr_genes'])
    num_pairs = len(smfish_genes)
    genes_1, genes_2 = [pair[0] for pair in smfish_genes], [pair[1] for pair in smfish_genes]
    gene_idxs_1, gene_idxs_2 = [dropseq_genes.index(g) for g in genes_1], [dropseq_genes.index(g) for g in genes_2]
    
    # Analysis setup
    num_samples = (len(NUMBER_OF_CELLS)-1) * len(METHODS) * NUM_TRIALS + len(METHODS)
    correlation_estimates = np.zeros((num_samples, num_pairs), dtype=np.float64)
    sample_details = []
    
    sample_idx = 0
        
    for num_cell in NUMBER_OF_CELLS:

        for trial in range(NUM_TRIALS if num_cell < 8000 else 1):

            # Iterate through the methods and calculate variance
            for method in METHODS:
                
                # if method == 'basics' and ((num_cell > 1000 and trial > 0) or (num_cell <= 1000 and trial > 1)):
                #     continue

                correlation_estimates[sample_idx, :] = estimate_correlations(method, num_cell, trial)
                sample_details.append((num_cell, trial+1, method))
                sample_idx += 1
                    
    metadata = pd.DataFrame(sample_details, columns=['num_cell', 'trial', 'method'])
    metadata.to_csv(DATA_PATH + 'smfish/correlation/sample_metadata.csv', index=False)
    np.savez_compressed(DATA_PATH + 'smfish/correlation/sample_correlations', correlations=correlation_estimates)

                                                         
                    
                
                
                
                
                

                
                


    