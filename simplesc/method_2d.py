import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
from statsmodels.stats.weightstats import DescrStatsW
import numpy as np
import itertools
import logging
from scipy.stats import multivariate_normal


def multivariate_normal_pmf(x, mean, cov, method):
	""" Multivariate normal PMF. """

	if method == 'pdf':

		return multivariate_normal.pdf(x, mean=mean, cov=cov)

	else:

		return \
			multivariate_normal.cdf(x + 0.5, mean=mean, cov=cov) - \
			multivariate_normal.cdf()


def calculate_px(x1, x2, mu, sigma, p, method):
	z_candidates = np.array(list(itertools.product(np.arange(x1, 25), np.arange(x2, 25))))
	return (
		stats.binom.pmf(x1, z_candidates[:, 0], p) * \
		stats.binom.pmf(x2, z_candidates[:, 1], p) * \
		multivariate_normal.pdf(z_candidates, mean=mu, cov=sigma)
		).sum()


def create_px_table(mu, sigma, p, method):

	px_table = np.zeros((20, 20))
	for i in range(20):
		for j in range(20):
			px_table[i][j] = calculate_px(i, j, mu, sigma, p, method)
	return px_table


def create_pz_table(mu, sigma, p, method):
	""" Returns a matrix M x M where rows indicate X and columns indicate Z """

	px_table = create_px_table(mu, sigma, p, method)

	table = []
	for x1 in range(15):
		for x2 in range(15):
			for z1 in range(x1, 15):
				for z2 in range(x2, 15):
					table.append((
						x1, x2, z1, z2,
						multivariate_normal.pdf([z1, z2], mean=mu, cov=sigma) * \
						stats.binom.pmf(x1, z1, p) * \
						stats.binom.pmf(x2, z2, p) / \
						px_table[x1, x2]))
	return pd.DataFrame(table, columns=['x1', 'x2', 'z1', 'z2', 'latent_weight'])


def get_parameters(observed, prob_table, initial_p_hat=0.1):
	""" Get the parameters of the Gaussian and dropout """

	data = pd.DataFrame(observed, columns=['x1', 'x2'])

	data = data.groupby(['x1', 'x2']).size().reset_index(name='count')
	data['observed_weight'] = data['count'] / len(observed)

	data = data.merge(
		prob_table,
		on=['x1', 'x2'],
		how='left')

	data['point_weight'] = data['observed_weight'] * data['latent_weight']
	stat_estimates = DescrStatsW(data[['z1', 'z2']], weights=data['point_weight'])
	p_estimate = initial_p_hat

	return stat_estimates.mean, stat_estimates.cov, p_estimate


def run_2d_em(observed,
	initial_p_hat=0.1,
	initial_mu_hat=[7, 7],
	initial_sigma_hat=[[5, 3], [3, 5]], 
	num_iter=400,
	method='pdf'):

	mu_hat = initial_mu_hat
	sigma_hat = initial_sigma_hat
	p_hat = initial_p_hat

	fitting_progress = []
	for itr in range(num_iter):
		print('Iteration: {}'.format(itr))

		fitting_progress.append((
			itr, 
			mu_hat[0], 
			mu_hat[1],
			sigma_hat[0][0], 
			sigma_hat[0][1],
			sigma_hat[1][0],
			sigma_hat[1][1],
			p_hat))

		# E step
		prob_table = create_pz_table(mu_hat, sigma_hat, p_hat, method=method)

		# M step
		mu_hat, sigma_hat, p_hat = get_parameters(observed, prob_table, initial_p_hat)

	fitting_progress = pd.DataFrame(
		fitting_progress, 
		columns=[
			'iteration', 
			'mu_hat_1', 
			'mu_hat_2',
			'sigma_hat_1_1', 
			'sigma_hat_1_2',
			'sigma_hat_2_1',
			'sigma_hat_2_2',
			'p_hat'])

	return mu_hat, sigma_hat, fitting_progress
