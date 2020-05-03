"""
	estimator.py
	
	Contains the estimators for the mean, variance, covariance, and the correlation. 
	
	Implements both the approximate hypergeometric and Poisson generative models. 
	
	If :data: is a tuple, it is assumed that it is a tuple of (value, count), both of which are 1D numpy arrays.
	If :data: is not a tuple, it is assumed to be a SciPy CSC sparse matrix.
"""

import scipy.sparse as sparse
import numpy as np
import time
import scipy as sp
import matplotlib.pyplot as plt


def _estimate_Nr(data):
	"""
		Estimate the average UMI count per cell
	"""


def _estimate_q_sq(data, q):
	"""
		Estimate q_sq from the data based on CV vs 1/mean.
	"""
	return

	
def _poisson_1d(data, n_obs=None):
	"""
		Estimate the variance using the Poisson noise process.
	"""
	
	if type(data) == tuple:
		obs_M1 = (data[0]*data[1]).sum(axis=0)/n_obs
		obs_M2 = (data[0]**2*data[1]).sum(axis=0)/n_obs
	else:
		obs_M1 = data.sum(axis=0).A1/n_obs
		obs_M2 = data.power(2).sum(axis=0).A1/n_obs

	mm_M1 = obs_M1
	mm_M2 = obs_M2 - obs_M1

	return mm_M1, mm_M2 - mm_M1**2


def _poisson_cov(data, n_obs, idx1=None, idx2=None):
	"""
		Estimate the covariance using the Poisson noise process between genes at idx1 and idx2.
	"""
	
	if type(data) == tuple:
		obs_M1 = (data[0]*data[2]).sum(axis=0)/n_obs
		obs_M2 = (data[1]*data[2]).sum(axis=0)/n_obs
		obs_MX = (data[0]*data[1]*data[2]).sum(axis=0)/n_obs
		cov = obs_MX - obs_M1*obs_M2

	else:
		overlap = set(idx1) & set(idx2)
		
		overlap_idx1 = np.array([1 if i in overlap else 0 for i in idx1])
		overlap_idx2 = np.array([1 if i in overlap else 0 for i in idx2])
		
		X, Y = data[:, idx1], data[:, idx2]
		prod = (X.T*Y).toarray()/X.shape[0]
		cov = prod - np.outer(X.mean(axis=0).A1, Y.mean(axis=0).A1)
		
		cov[overlap_idx1[:, np.newaxis], overlap_idx2] -= X[:, overlap_idx1].sum(axis=0).A1/n_obs
	
	return cov


def _hyper_mean(data, q, q_sq):
	"""
		Estimate the mean using the (approximate) hypergeometric noise process.
	"""
	
	if type(data) == tuple:
		return (data[0]*data[1]).sum(axis=0)/q
	else:
		return data.mean(axis=0).A1/q


def _hyper_1d(data, n_obs, q, q_sq):
	"""
		Estimate the variance using the (approximate) hypergeometric noise process.
	"""
	
	if type(data) == tuple:
		obs_M1 = (data[0]*data[1]).sum()/n_obs
		obs_M2 = (data[0]**2*data[1]).sum()/n_obs
	
	else:
		obs_M1 = data.mean(axis=0).A1
		obs_M2 = data.power(2).mean(axis=0).A1
		
	mm_M1 = obs_m1/q
	mm_M2 = (obs_M2 + (q_sq/q)*obs_M1 - obs_M1)/q_sq

	return mm_M1, mm_M2 - mm_M1**2


def _hyper_cov(data, idx1, idx2, q, q_sq):
	"""
		Estimate the covariance using the (approximate) hypergeometric noise process.
	"""

	if type(data) == tuple:
		obs_M1 = (data[0]*data[2]).sum(axis=0)/n_obs
		obs_M2 = (data[1]*data[2]).sum(axis=0)/n_obs
		obs_MX = (data[0]*data[1]*data[2]).sum(axis=0)/n_obs
		cov = obs_MX/q_sq - obs_M1*obs_M2/q**2

	else:
		overlap = set(idx1) & set(idx2)
		overlap_idx1 = np.array([1 if i in overlap else 0 for i in idx1])
		overlap_idx2 = np.array([1 if i in overlap else 0 for i in idx2])
		
		X, Y = data[:, idx1], data[:, idx2]
		obs_MX = (X.T*Y).toarray()/n_obs
		obs_M1_1 = X.sum(axis=0).A1/n_obs
		obs_M1_2 = Y.sum(axis=0).A1/n_obs
		mm_MX = obs_MX/q_sq 
		mm_MX[overlap_idx1[:, np.newaxis], overlap_idx2] += ((q_sq/q)*obs_M1_1[overlap_idx1]-obs_M1_1[overlap_idx1])/q_sq

		cov = mm_MX - np.outer(obs_M1_1, obs_M1_2)/q**2
			
	return cov


def _corr_from_cov(cov, var_1, var_2):
	"""
		Convert the estimation of the covariance to the estimation of correlation.
	"""

	corr = cov / np.outer(np.sqrt(var_1), np.sqrt(var_2))
	corr[corr < -1] = np.nan
	corr[corr > 1] = np.nan
	
	return corr