import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from statsmodels.stats.weightstats import DescrStatsW
import numpy as np


def gaussian_pmf(x, mu, sigma, method):
	""" Gaussian PMF. x may be an array. """

	if method == 'pmf':
		return stats.norm.cdf(x+0.5, mu, sigma) - stats.norm.cdf(x-0.5, mu, sigma)
	elif method == 'pdf':
		return stats.norm.pdf(x, mu, sigma)


def create_px_table(mu, sigma, p, method):
	return np.array([
		(gaussian_pmf(np.arange(x, 20), mu, sigma, method) * stats.binom.pmf(x, np.arange(x, 20), p)).sum()
		for x in range(30)])


def create_pz_table(mu, sigma, p, method):
	""" Returns a matrix M x M where rows indicate X and columns indicate Z """

	px_table = create_px_table(mu, sigma, p, method)
	table = np.zeros(shape=(20, 20))
	for x in range(20):
		for z in range(x, 20):
			table[x, z] = gaussian_pmf(z, mu, sigma, method) * stats.binom.pmf(x, z, p) / px_table[x]
	return table


def get_parameters(observed, prob_table, initial_p_hat=0.1):
	""" Get the parameters of the Gaussian and dropout """
	data = pd.DataFrame()
	data['observed'] = observed
	data = data.groupby('observed').size().reset_index(name='count')
	data['observed_weight'] = data['count'] / len(observed)
	data = data.merge(
	pd.concat(
		[pd.DataFrame(
			np.concatenate(
				[np.ones(20-x).reshape(-1, 1)*x, 
				np.arange(x, 20).reshape(-1,1),
				prob_table[x, x:].reshape(-1, 1)], axis=1), 
			columns=['observed', 'latent', 'latent_weight']) for x in range(20)]),
		on='observed', 
		how='left')
	data['point_weight'] = data['observed_weight'] * data['latent_weight']
	data['p_estimates'] = (data['observed'] / data['latent'] * data['point_weight']).fillna(0.0).replace(np.inf, 0.0)
	p_estimate = initial_p_hat #min(max(data['p_estimates'].sum(), 0.05), 0.15)
	stat_estimates = DescrStatsW(data['latent'], weights=data['point_weight'])
	return stat_estimates.mean, np.sqrt(stat_estimates.var), p_estimate


def run_1d_em(observed,
	initial_p_hat=0.1,
	initial_mu_hat=10,
	initial_sigma_hat=5, 
	method='pdf',
	num_iter=400):
	""" Run the 1D expectation maximization algorithm. """

	p_hat = initial_p_hat
	mu_hat = initial_mu_hat
	sigma_hat = initial_sigma_hat

	fitting_progress = []

	for itr in range(num_iter):

		fitting_progress.append((
			itr, 
			mu_hat, 
			sigma_hat, 
			p_hat))

		# E step
		prob_table = create_pz_table(
			mu_hat, 
			sigma_hat, 
			p_hat,
			method=method)

		# M step
		mu_hat, sigma_hat, p_hat = get_parameters(
			observed, 
			prob_table,
			initial_p_hat=initial_p_hat)

	fitting_progress = pd.DataFrame(
		fitting_progress, 
		columns=[
			'iteration', 
			'mu_hat', 
			'sigma_hat', 
			'p_hat'])

	return mu_hat, sigma_hat, fitting_progress


def reconstruct_distribution(mu_hat, sigma_hat, num_cells=10000):
	""" Reconstruct the inferred underlying distribution. """
	
	return np.clip(
		np.round(
			np.random.normal(
				mu_hat, 
				sigma_hat, 
				size=num_cells)), 
		a_min=0, 
		a_max=100).astype(np.int64)

