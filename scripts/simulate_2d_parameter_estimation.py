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
	mu = [3, 5]
	sigma = [[3, 2],[2, 2]]

	continuous_gaussian= multivariate_normal.rvs(mu, sigma, size=num_cells)

	ground_truth_counts = np.clip(
		np.round(continuous_gaussian),
		a_min=0, 
		a_max=100).astype(np.int64)

	observed = np.random.binomial(n=ground_truth_counts, p=p)

	mu_hat, sigma_hat, progress = md2.run_2d_em(observed)

	print(mu_hat)
	print(sigma_hat)

	progress.to_csv('/netapp/home/mincheol/param_est_data/simulated_2d_em_progress.csv', index=False)