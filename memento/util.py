import scipy.stats as stats
import numpy as np
import time
import itertools
import scipy as sp
from statsmodels.stats.multitest import fdrcorrection

def _select_cells(adata, group):
	""" Slice the data horizontally. """

	cell_selector = (adata.obs['memento_group'] == group).values
	
	return adata.X[cell_selector, :].tocsc()


def _get_gene_idx(adata, gene_list):
	""" Returns the indices of each gene in the list. """

	return np.array([np.where(adata.var.index == gene)[0][0] for gene in gene_list]) # maybe use np.isin


def _fdrcorrect(pvals):
	"""
		Perform FDR correction with nan's.
	"""

	fdr = np.ones(pvals.shape[0])
	_, fdr[~np.isnan(pvals)] = fdrcorrection(pvals[~np.isnan(pvals)])
	return fdr


def density_scatterplot(a,b, s=1, cmap='Reds', kde=None):
    # Calculate the point density
    condition = np.isfinite(a) & np.isfinite(b)
    x = a[condition]
    y = b[condition]
    xy = np.vstack([x,y])
    z = stats.gaussian_kde(xy, bw_method=kde)(xy)
    print(z)
    plt.scatter(x, y, c=z, s=s, edgecolor='', cmap=cmap)
    

def robust_correlation(a, b):
    
    condition = (np.isfinite(a) & np.isfinite(b))
    x = a[condition]
    y = b[condition]
    
    return stats.spearmanr(x,y)

def robust_linregress(a, b):
    
    condition = (np.isfinite(a) & np.isfinite(b))
    x = a[condition]
    y = b[condition]
    
    print(x.min())
    
    return stats.linregress(x,y)

def robust_hist(x, **kwargs):
    
    condition = np.isfinite(x)
    plt.hist(x[condition], **kwargs)