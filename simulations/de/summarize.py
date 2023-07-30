import scanpy as sc
import pandas as pd
import numpy as np
import functools
import matplotlib.pyplot as plt

data_path = '/data_volume/memento/simulation/'


if __name__ == '__main__':
    
    # Read anndata object and setup results
    adata = sc.read(data_path + 'de/anndata.h5ad')

    # Set up results holder
    results = {}

    # Read edger
    results['edger_lrt'] = pd.read_csv(data_path + 'de/edger_lrt.csv', index_col=0)[['logFC', 'PValue', 'FDR']]
    results['edger_qlft'] = pd.read_csv(data_path + 'de/edger_qlft.csv', index_col=0)[['logFC', 'PValue', 'FDR']]

    # Read memento
    results['memento'] = pd.read_csv(data_path + 'de/memento.csv', index_col=0)[['coef', 'pval', 'fdr']]
    results['memento_wls'] = pd.read_csv(data_path + 'de/memento_wls.csv', index_col=0)[['coef', 'pval', 'fdr']]


    # Read t test
    results['t'] = pd.read_csv(data_path + 'de/t.csv', index_col=0)[['coef', 'pval', 'fdr']]

    # Get overlapping genes
    genes = set(adata.var.index)
    for method, result in results.items():
        results[method] = result.join(adata.var[['is_de']],how='inner')
        genes &= set(result.index)
    genes = list(genes)
    for method, result in results.items():
        results[method] = result.loc[genes]
        results[method].columns = ['coef', 'pval', 'fdr', 'is_de']

    # Print some statistics
    n=12
    thresholds = {
        'memento':np.linspace(0.001, 0.2, n),
        't':np.linspace(0.001, 0.2, n),
        'edger_lrt':np.linspace(0.01, 0.7, n),
        'edger_qlft':np.linspace(0.01, 0.6, n-2)       
    }
    check_thresh = 0.05
    plt.figure()
    for method, result in results.items():

        tpr = (result.query('is_de')['pval'] < check_thresh).mean()
        fpr = (result.query('~is_de')['pval'] < check_thresh).mean()
        print(f'{method} - fpr: {fpr} - tpr : {tpr} - thresh : {check_thresh}')
                
    
    # Make some figures
    
    
