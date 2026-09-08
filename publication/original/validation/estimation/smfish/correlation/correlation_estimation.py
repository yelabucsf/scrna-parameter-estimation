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
NUM_TRIALS = 20
# NUMBER_OF_CELLS = [5000]
NUMBER_OF_CELLS = [500, 1000, 5000, 8000]
# NUMBER_OF_CELLS = [500, 1000, 5000]

MIN_MEAN_THRESH = 0.01

# METHODS = ['hypergeometric','poisson', 'saver' ,'naive']
METHODS = ['naive', 'saver', 'poisson', 'hypergeometric', 'scvi']

NUM_SAMPLES = NUM_TRIALS*3*len(NUMBER_OF_CELLS)
NUM_SAMPLES += NUM_TRIALS*2 + 1 # for saver
NUM_SAMPLES += NUM_TRIALS*2 + 1 # for scvi
NUM_SAMPLES += len(METHODS) # for using all cells


def estimate_correlations(method, nc, trial): # x_param used for scaling for basics
    
    captured_data = sc.read_h5ad(DATA_PATH + f'smfish/variance/{num_cell}_{trial}.h5ad')
    data = captured_data.X
    size_factor = data.sum(axis=1).A1
    obs_mean = data.mean(axis=0).A1
    # size_factor = np.ones(captured_data.shape[0]) # no normalization
    
    if method == 'naive':
        
        mat = data.toarray()/size_factor.reshape(-1,1)
        correlations = np.array([stats.pearsonr( mat[:, i1], mat[:, i2] )[0] for i1,i2 in zip(gene_idxs_1, gene_idxs_2)])
        correlations[(obs_mean[gene_idxs_1]<MIN_MEAN_THRESH) | (obs_mean[gene_idxs_2]<MIN_MEAN_THRESH)] = np.nan

        return correlations

    elif method == 'poisson':
        
        est = memento.estimator.RNAPoisson()
        sf = est.estimate_size_factor(data, shrinkage=0.6, filter_mean_thresh=0.07, trim_percent=0.05)
        correlations = est.correlation(data, sf, gene_idxs_1, gene_idxs_2)
        correlations[(obs_mean[gene_idxs_1]<MIN_MEAN_THRESH) | (obs_mean[gene_idxs_2]<MIN_MEAN_THRESH)] = np.nan

        return correlations
    
    elif method == 'hypergeometric':
        
        est = memento.estimator.RNAHypergeometric(0.01485030176341905)
        sf = est.estimate_size_factor(data, shrinkage=0.6, filter_mean_thresh=0.07, trim_percent=0.05)
        correlations = est.correlation(data, sf, gene_idxs_1, gene_idxs_2)
        correlations[(obs_mean[gene_idxs_1]<MIN_MEAN_THRESH) | (obs_mean[gene_idxs_2]<MIN_MEAN_THRESH)] = np.nan

        return correlations
    
    elif method == 'saver': # just load the values already calculated from saver R script
        
        print('Getting saver results')
        saver_correlations = pd.read_csv(DATA_PATH + f'smfish/correlation/{nc}_{trial}_corr2.csv', index_col=0) #corr2 is from cor.genes function
        if nc < 8000:
            gene_mapping = dict(zip([f'gene{i}' for i in range(len(dropseq_genes))], dropseq_genes))
            remapped_names = [gene_mapping[n] for n in saver_correlations.index]
            saver_correlations.index = remapped_names
            saver_correlations.columns = remapped_names
        correlations = np.array([saver_correlations.loc[a,b] for a,b in smfish_estimates['corr_genes']])
        correlations[(obs_mean[gene_idxs_1]<MIN_MEAN_THRESH) | (obs_mean[gene_idxs_2]<MIN_MEAN_THRESH)] = np.nan

        return correlations
    
    elif method == 'scvi': # Just load the values already calculated from scvi script (run on GPU instance)
        
        scvi_correlations = pd.read_csv(DATA_PATH + f'smfish/correlation/{nc}_{trial}_scvi_corr.csv', index_col=0)
        scvi_genes = scvi_correlations.index.tolist()
        correlations = np.array([scvi_correlations.loc[a,b] for a,b in smfish_estimates['corr_genes']])
        correlations[(obs_mean[gene_idxs_1]<MIN_MEAN_THRESH) | (obs_mean[gene_idxs_2]<MIN_MEAN_THRESH)] = np.nan
        return correlations
    
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
    num_samples = NUM_SAMPLES
    correlation_estimates = np.zeros((num_samples, num_pairs), dtype=np.float64)
    sample_details = []
    
    sample_idx = 0
        
    for num_cell in NUMBER_OF_CELLS:

        for trial in range(NUM_TRIALS if num_cell < 8000 else 1):

            # Iterate through the methods and calculate variance
            for method in METHODS:
                
                if method in ['saver', 'scvi'] and( trial > 0 and num_cell > 1000):
                    continue

                correlation_estimates[sample_idx, :] = estimate_correlations(method, num_cell, trial)
                sample_details.append((num_cell, trial+1, method))
                sample_idx += 1
                    
    metadata = pd.DataFrame(sample_details, columns=['num_cell', 'trial', 'method'])
    metadata.to_csv(DATA_PATH + 'smfish/correlation/sample_metadata.csv', index=False)
    np.savez_compressed(DATA_PATH + 'smfish/correlation/sample_correlations', correlations=correlation_estimates)

                                                         
                    
                
                
                
                
                

                
                


    