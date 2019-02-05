import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
from statsmodels.stats.weightstats import DescrStatsW
import numpy as np
import itertools
import logging
from scipy.stats import multivariate_normal


def lognormal_pdf_2d(x, mean, cov):
    
    denom = x[0]*x[1] if type(x) == list else (x[:, 0] * x[:, 1])
    return np.nan_to_num(stats.multivariate_normal.pdf(np.log(x), mean=mean, cov=cov)/denom)


def fit_multivariate_distribution(data, d1_col='z1', d2_col='z2', weights_col='point_weight', method='lognormal'):

	if method == 'normal':

		parameters = DescrStatsW(data[[d1_col, d2_col]], weights=data[weights_col])

	elif method == 'lognormal':

		ln_data = data.query('{} > 0 & {} > 0'.format(d1_col, d2_col))
		parameters = DescrStatsW(np.log(ln_data[[d1_col, d2_col]]), weights=ln_data[weights_col])

	temp = DescrStatsW(data[[d1_col, d2_col]], weights=data[weights_col])
	ln_diag = temp.cov[0, 1]
	direct = np.log(ln_diag / np.exp(1.1 + 1.5 + (.36+.16)/2) + 1)
	print(parameters.cov[0, 1], direct)

	return [1.1, 1.5], np.array([[.36, direct], [direct, 0.16]])


def multivariate_pdf(x, mean, cov, method='lognormal'):
	""" Multivariate normal PMF. """

	if method == 'normal':

		return multivariate_normal.pdf(x, mean=mean, cov=cov)

	if method == 'lognormal':

		return lognormal_pdf_2d(x, mean=mean, cov=cov)


def calculate_px(x1, x2, mu, sigma, p, method, max_count=21):
	z_candidates = np.array(list(itertools.product(np.arange(x1, max_count), np.arange(x2, max_count))))
	return (
		stats.binom.pmf(x1, z_candidates[:, 0], p) * \
		stats.binom.pmf(x2, z_candidates[:, 1], p) * \
		multivariate_pdf(z_candidates, mean=mu, cov=sigma, method=method)
		).sum()


def create_px_table(mu, sigma, p, method, max_count=21):

	px_table = np.zeros((max_count, max_count))
	for i in range(max_count):
		for j in range(max_count):
			px_table[i][j] = calculate_px(i, j, mu, sigma, p, method)
	return px_table


def create_pz_table(mu, sigma, p, method, max_count=20):
	""" Returns a matrix M x M where rows indicate X and columns indicate Z """

	px_table = create_px_table(mu, sigma, p, method)

	table = []
	for x1 in range(max_count):
		for x2 in range(max_count):
			for z1 in range(x1, max_count):
				for z2 in range(x2, max_count):
					table.append((
						x1, x2, z1, z2,
						multivariate_pdf([z1, z2], mean=mu, cov=sigma, method=method) * \
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

	mu, sigma = fit_multivariate_distribution(data, d1_col='z1', d2_col='z2', weights_col='point_weight')
	p = initial_p_hat

	return mu, sigma, p


def run_2d_em(observed,
	initial_p_hat=0.1,
	initial_mu_hat=[1.1, 1.5],
	initial_sigma_hat=[[0.36, 0.01], [0.01, 0.16]], 
	num_iter=400,
	method='lognormal'):

	mu_hat = initial_mu_hat
	sigma_hat = initial_sigma_hat
	p_hat = initial_p_hat

	fitting_progress = []
	for itr in range(num_iter):
		print('Iteration: {}, mu={}, sigma={}'.format(itr, mu_hat, sigma_hat))

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
