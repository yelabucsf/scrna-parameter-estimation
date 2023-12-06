# simulate_mean_estimation.py

# Authors: Min Cheol Kim

# This script generates the figure for mean comparisons, comparing the naive mean estimator and the memento estimator. 

import scanpy as sc
import matplotlib.pyplot as plt

import sys
sys.path.append('/home/ubuntu/Github/memento/')
import memento


DATA_PATH = '/home/ubuntu/Data/'
CELL_TYPE = 'CD4 T cells - ctrl'

NUM_TRIALS = 1
NUM_METHODS = 5
CAPTURE_EFFICIENCIES = [0.1]
NUMBER_OF_CELLS = [30]


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
    
    adata = sc.read(data_path + 'interferon_filtered.h5ad')
    adata = adata[adata.obs.cell_type == CELL_TYPE]
    data = adata.X.copy()
    relative_data = data.toarray()/data.sum(axis=1)
    
    x_param, z_param, Nc, good_idx = memento.simulate.extract_parameters(adata.X, q=q, min_mean=0.01)
    
    return x_param, z_param, Nc


def simulate_data(n_cells, z_param, Nc):
    """ Generates simulated data. """
    
    true_data = memento.simulate.simulate_transcriptomes(n_cells=n_cells, means=z_param[0], variances=z_param[1], Nc=Nc, norm_cov='uncorrelated')
    true_data[true_data < 0] = 0
    
    qs, captured_data = memento.simulate.capture_sampling(true_data, q, q_sq=None)
    captured_data = sp.sparse.csr_matrix(captured_data)
    
    return true_data, captured_data


def estimate_means(method):
    
    
    if method == 'naive':
        
    elif method == 'pb':
        
    elif method == 'good_turing':
        
    else:
        
        raise 'Not implemented!'
        

if __name__ == '__main__':
    
    # Extract gene expression parameters to be used across all simulations
    x_param, z_param, Nc = get_simulation_parameters(q=0.07)
    true_mean = x_param[0]
    
    # Simulation setup
    num_simulations = len(CAPTURE_EFFICIENCIES) * len(NUMBER_OF_CELLS) * NUM_TRIALS
    mean_estimates = np.array()
    
    for q in CAPTURE_EFFICIENCIES:
        
        for num_cell in NUMBER_OF_CELLS:
            
            for trial in NUM_TRIALS:
                
                true_data, captured_data = simulate_data(num_cell, z_param, Nc)
                
                
                


    