"""
	sc_estimator.py
	This file contains code for fitting 1d and 2d lognormal and normal parameters to scRNA sequencing count data.
"""


import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
from statsmodels.stats.weightstats import DescrStatsW
import numpy as np
import itertools
import logging
from scipy.stats import multivariate_normal


class SingleCellEstimator(object):
	"""
		SingleCellEstimator is the class for fitting univariate and bivariate single cell data. 
	"""

	def __init__(self, adata, p=0.1):

		self.anndata = adata
		self.genes = adata.var.index
		self.barcodes = adata.obs.index
		self.p = p


	def compute_1d_params(self, gene):