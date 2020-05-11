import scipy.stats as stats
import numpy as np
import time
import itertools
import scipy as sp
from statsmodels.stats.multitest import fdrcorrection

def _select_cells(adata, group):
	""" Slice the data horizontally. """

	cell_selector = (adata.obs['scmemo_group'] == group).values
	
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