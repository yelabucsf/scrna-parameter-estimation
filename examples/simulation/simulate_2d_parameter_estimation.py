import method_2d as md2
import pandas as pd
import sys
import os
from scipy.stats import multivariate_normal
import numpy as np

class ForceIOStream:
	def __init__(self, stream):
		self.stream = stream

	def write(self, data):
		self.stream.write(data)
		self.stream.flush()
		if not self.stream.isatty():
			os.fsync(self.stream.fileno())

	def __getattr__(self, attr):
		return getattr(self.stream, attr)


sys.stdout = ForceIOStream(sys.stdout)
sys.stderr = ForceIOStream(sys.stderr)

if __name__ =='__main__':

	# True Parameters
	num_cells = 100000
	p = 0.1
	mu = [1.1, 1.5]
	sigma = [[0.6**2, 0.1],[0.1, 0.4**2]]

	print('True params: mu={}, sigma={}'.format(mu, sigma))

	continuous_gaussian = multivariate_normal.rvs(mu, sigma, size=num_cells)
	continuous = np.exp(continuous_gaussian)

	ground_truth_counts = np.round(continuous).astype(np.int64)

	observed = np.random.binomial(n=ground_truth_counts, p=p)

	mu_hat, sigma_hat, progress = md2.run_2d_em(
		observed, 
		num_iter=200)

	print(mu_hat)
	print(sigma_hat)

	#progress.to_csv('/netapp/home/mincheol/simulated_2d_em_progress.csv', index=False)