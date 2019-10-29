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
	design_matrix = pd.get_dummies(df, columns=['batch']).values

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

	return stats.binom.rvs(true_counts, qs)

