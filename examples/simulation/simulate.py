"""
	simulate.py

	Functions for simulating various gene expression patterns with beta-binomial dropout.
"""

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scipy as sp
from mixedvines.copula import Copula, GaussianCopula, ClaytonCopula, FrankCopula
from mixedvines.mixedvine import MixedVine

def convert_params(mu, theta):
	"""
	Convert mean/dispersion parameterization of a negative binomial to the ones scipy supports

	See https://en.wikipedia.org/wiki/Negative_binomial_distribution#Alternative_formulations
	"""
	r = theta
	var = mu + 1 / r * mu ** 2
	p = (var - mu) / var
	return r, 1 - p


def simulate_true_counts(
	N,
	mean_coef,
	var_coef,
	num_levels = 3,
	p=1):
	"""
		:mean_coef: First element is the coefficient for the treatment. Log-linear function of the mean.
		:var_coef: First element is the coefficient for the treatment. Log-linear function of the variance.
		:distribution: The distribution that the simulation is based on.
		:num_levels: number of levels of the treatment (e.g., genotypes are num_levels=3)

		Assumes that the batch and the covariate variables are independent.
		Assumes that all batches have the same number of cells.
		Assumes that the levels of the condition has same number of cells.
	"""

	# Assign the condition
	condition_cov = np.random.randint(num_levels, size=N)

	# Assign the batch
	batch_cov = np.random.choice(['batch-' + str(i) for i in range(1, mean_coef.shape[0])], size=N)

	# Create the design matrix
	df = pd.DataFrame()
	df['condition'] = condition_cov
	df['batch'] = batch_cov
	df = pd.get_dummies(df, columns=['batch'], drop_first=True)
	df['intercept'] = 1

	design_matrix = df.values
	# Get the mean and variances
	mean_vector = np.exp(design_matrix@mean_coef.reshape(-1, 1))
	var_vector = np.exp(design_matrix@(var_coef + mean_coef).reshape(-1, 1))

	# Sample from a negative binomial
	response = stats.nbinom.rvs(*convert_params(mean_vector, 1/((var_vector-mean_vector)/mean_vector**2)))

	return design_matrix, response


def simulate_dropout(
	true_counts,	
	q,
	q_sq):
	""" 
		:true_counts: True counts
		:q: first moment of the dropout probability
		:q_sq: second moment of the dropout probability

		Simulate the beta-binomial dropout.
	"""

	m = q
	v = q_sq - q**2
	alpha = m*(m*(1-m)/v - 1)
	beta = (1-m)*(m*(1-m)/v - 1)
	qs = stats.beta.rvs(alpha, beta, size=true_counts.shape)
	print('hi')
	return stats.binom.rvs(true_counts, qs)


def simulate_correlated_transcriptome(num_cells, num_genes, num_eig, p=1.5):
	"""
		Simulate two groups of cells.
	"""

	while(True):
		try:
			# Get the correlation matrix
			eigen_values = stats.poisson.rvs(50, size=num_eig)
			eigen_values = eigen_values/eigen_values.sum()*num_genes
			eigen_values = np.concatenate([eigen_values, np.zeros(num_genes-num_eig)])
			corr = stats.random_correlation.rvs(eigen_values)
			corr[corr > 0.99] = 0.95
			corr[corr < -0.99] = -0.95
			break
		except:
			continue

	# Get the means
	means = stats.lognorm.rvs(s=1, scale=15, size=num_genes)

	# Get the variances
	mean_independent_variance_1 = stats.lognorm.rvs(s=0.5, scale=15, size=num_genes)

	# Get the variances for the second group
	mean_independent_variance_2 = mean_independent_variance_1
	mean_independent_variance_2[:int(num_genes/2)] = mean_independent_variance_2[:int(num_genes/2)] + \
		stats.lognorm.rvs(s=0.5, scale=4, size=int(num_genes/2))
	variances_1 = mean_independent_variance_1 * (means**p)
	variances_2 = mean_independent_variance_2 * (means**p)

	# Get dispersions
	dispersions_1 = (variances_1-means)/means**2
	dispersions_2 = (variances_2-means)/means**2

	# Get the thetas
	thetas_1 = 1/dispersions_2
	thetas_2 = 1/dispersions_2

	# Set up vines
	vine_1 = MixedVine(num_genes)
	vine_2 = MixedVine(num_genes)

	# Set up marginals
	for i in range(num_genes):
		vine_1.set_marginal(i, stats.nbinom(*convert_params(means[i], dispersions_1[i])))
		vine_2.set_marginal(i, stats.nbinom(*convert_params(means[i], dispersions_2[i])))

	# Set up copulas
	for i in range(1, num_genes+1):
		for j in range(num_genes-i):
			vine_1.set_copula(i, j, GaussianCopula(corr[i, j]))
			vine_2.set_copula(i, j, GaussianCopula(corr[i, j]))

	data_1 = vine_1.rvs(num_cells)
	data_2 = vine_2.rvs(num_cells)

	metadata = {
		'means':means,
		'var1':variances_1,
		'var2':variances_2,
		'mean_ind_var1':mean_independent_variance_1,
		'mean_ind_var2':mean_independent_variance_2,
		'disp1':dispersions_1,
		'disp2':dispersions_2,
		'corr':corr
	}

	return metadata,data_1, data_2



