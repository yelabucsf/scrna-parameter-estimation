import pandas as pd
import matplotlib.pyplot as plt
import scanpy.api as sc
import scipy as sp
import itertools
import numpy as np
import scipy.stats as stats
from scipy.integrate import dblquad
import seaborn as sns
from statsmodels.stats.multitest import fdrcorrection
from sklearn.datasets import make_spd_matrix
import imp
pd.options.display.max_rows = 999
pd.set_option('display.max_colwidth', -1)
import pickle as pkl
import time


def convert_params(mu, theta):
	"""
	Convert mean/dispersion parameterization of a negative binomial to the ones scipy supports

	See https://en.wikipedia.org/wiki/Negative_binomial_distribution#Alternative_formulations
	"""
	r = theta
	var = mu + 1 / r * mu ** 2
	p = (var - mu) / var
	return r, 1 - p


def simulate_transcriptomes(
	n_cells, 
	n_genes, 
	correlated=False, 
	mv_mean=[1, 2], 
	mv_cov=[[4, 0], [0, 0.4]],
	log_means=None,
	log_res_var=None,
	norm_cov=None
	):
		
	# Define the parameters for the marginals
	if log_means is None and log_res_var is None:
		params = stats.multivariate_normal.rvs(mean=mv_mean, cov=mv_cov, size=n_genes)
		means, residual_variances = np.exp(params[:, 0]), np.exp(params[:, 1])
	else:
		means, residual_variances = np.exp(log_means), np.exp(log_res_var)
	variances = means*residual_variances
	dispersions = (variances - means)/means**2
	dispersions[dispersions < 0] = 1e-5
	thetas = 1/dispersions
	
	if not correlated:
		return stats.nbinom.rvs(*convert_params(means, thetas), size=(n_cells, n_genes))
	
	# Define parameters for the copula
	norm_mean = np.random.random(n_genes)
	norm_cov = make_spd_matrix(n_genes) if norm_cov is None else norm_cov
	norm_var = np.diag(norm_cov)
	corr = norm_cov / np.outer(np.sqrt(norm_var), np.sqrt(norm_var))
	
	# Sample from the copula
	gaussian_variables = stats.multivariate_normal.rvs(mean=norm_mean, cov=norm_cov, size=n_cells)
	uniform_variables = stats.norm.cdf(gaussian_variables, loc=norm_mean, scale=np.sqrt(norm_var))
	nbinom_variables = stats.nbinom.ppf(uniform_variables, *convert_params(means, thetas))
	
	return nbinom_variables.astype(int)


def capture_sampling(transcriptomes, q, q_sq):
	
	m = q
	v = q_sq - q**2
	alpha = m*(m*(1-m)/v - 1)
	beta = (1-m)*(m*(1-m)/v - 1)
	qs = stats.beta.rvs(alpha, beta, size=transcriptomes.shape[0])
	gen = np.random.Generator(np.random.PCG64(42343))
	
	captured_transcriptomes = []
	for i in range(transcriptomes.shape[0]):
		captured_transcriptomes.append(
			gen.multivariate_hypergeometric(transcriptomes[i, :], np.round(qs[i]*transcriptomes[i, :].sum()).astype(int))
		)
	
	return qs, np.vstack(captured_transcriptomes)


def sequencing_sampling(transcriptomes):
	
	observed_transcriptomes = np.zeros(transcriptomes.shape)
	num_molecules = transcriptomes.sum()
	print(num_molecules)
	
	for i in range(n_cells):
		for j in range(n_genes):
			
			observed_transcriptomes[i, j] = (stats.binom.rvs(n=int(num_reads), p=1/num_molecules, size=transcriptomes[i, j]) > 0).sum()
			
	return observed_transcriptomes