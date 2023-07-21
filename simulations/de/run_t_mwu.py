import pandas as pd
import matplotlib.pyplot as plt
import scanpy as sc
import scipy as sp
import itertools
import numpy as np
import scipy.stats as stats
from scipy.integrate import dblquad
from statsmodels.stats.multitest import fdrcorrection
import random
import statsmodels.api as sm


data_path = '/data_volume/memento/simulation/'


if __name__ == '__main__':
    
    adata = sc.read(data_path + 'de/norm_anndata.h5ad')
    A_data, B_data = adata[adata.obs['condition'] == 'ctrl'].X.toarray(), adata[adata.obs['condition'] == 'stim'].X.toarray()
    
    result = pd.DataFrame(index=adata.var.index)
    result['coef'] =A_data.mean(axis=0) - B_data.mean(axis=0)
    _, result['pval'] = stats.ttest_ind(A_data, B_data, equal_var=True)
    result = result.fillna(1.0)
    _, result['fdr'] = fdrcorrection(result['pval'])
    
    result.to_csv(data_path + 'de/t.csv')
    
    print('t test successful')
