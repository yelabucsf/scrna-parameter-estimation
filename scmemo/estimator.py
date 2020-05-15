"""
	estimator.py
	
	Contains the estimators for the mean, variance, covariance, and the correlation. 
	
	Implements both the approximate hypergeometric and Poisson generative models. 
	
	If :data: is a tuple, it is assumed that it is a tuple of (value, count), both of which are 1D numpy arrays.
	If :data: is not a tuple, it is assumed to be a SciPy CSC sparse matrix.
"""

import scipy.sparse as sparse
import scipy.stats as stats
import numpy as np
import time
import scipy as sp
import matplotlib.pyplot as plt
from sklearn import linear_model


def _estimate_Nr(data):
	"""
		Estimate the average UMI count per cell
	"""


def _estimate_q_sq(data, q):
	"""
		Estimate q_sq from the data based on CV vs 1/mean.
	"""
	return

def _estimate_size_factor(data):
	"""Calculate the size factor
	
	Args: 
		data (AnnData): the scRNA-Seq CG (cell-gene) matrix.
		
	Returns:
		size_factor((Nc,) ndarray): the cell size factors.
	"""
	X=data
	Nrc = np.array(X.sum(axis=1)).reshape(-1)
	Nr = Nrc.mean()
	size_factor = Nrc/Nr
	return size_factor


def _fit_mv_regressor(mean, var):
	"""
		Perform linear regression of the variance against the mean.
		
		Uses the RANSAC algorithm to choose a set of genes to perform 1D linear regression.
	"""
	
	cond = (mean > 0) & (var > 0)
	m, v = np.log(mean[cond]), np.log(var[cond])
	slope, inter, _, _, _ = stats.linregress(m, v)
	
	return [slope, inter]


def _residual_variance(mean, var, mv_fit):
	
	cond = (mean > 0)
	rv = np.zeros(mean.shape)*np.nan
# 	rv[cond] = var[cond] / (np.exp(mv_fit[1])*mean[cond]**mv_fit[0])
	rv[cond] = var[cond]/mean[cond]
	
	return rv


def _poisson_1d(data, n_obs, size_factor=None, n_umi=1):
	"""
		Estimate the variance using the Poisson noise process.
		
		If :data: is a tuple, :cell_size: should be a tuple of (inv_sf, inv_sf_sq). Otherwise, it should be an array of length data.shape[0].
	"""
	if type(data) == tuple:
		size_factor = size_factor if size_factor is not None else (1, 1)
		mm_M1 = (data[0]*data[1]*size_factor[0]).sum(axis=0)/n_obs
# 		mm_M2 = (data[0]**2*data[1]*size_factor[1]).sum(axis=0)/n_obs
		mm_M2 = (data[0]**2*data[1]*size_factor[1] - data[0]*data[1]*size_factor[1]).sum(axis=0)/n_obs
	else:
		row_weight = (1/size_factor).reshape([1, -1]) if size_factor is not None else np.ones(data.shape[0])
		mm_M1 = sparse.csc_matrix.dot(row_weight, data).ravel()/n_obs
# 		mm_M2 = sparse.csc_matrix.dot(row_weight**2, data.power(2)).ravel()/n_obs

		mm_M2 = sparse.csc_matrix.dot(row_weight**2, data.power(2)).ravel()/n_obs - sparse.csc_matrix.dot(row_weight**2, data).ravel()/n_obs
	
	mm_mean = mm_M1/n_umi
	mm_var = (mm_M2 - mm_M1**2)/n_umi**2
# 	mm_var = np.clip(mm_var, a_min=1e-20, a_max=np.inf)

	return [mm_mean, mm_var]


def _poisson_cov(data, n_obs, size_factor, idx1=None, idx2=None, n_umi=1):
	"""
		Estimate the covariance using the Poisson noise process between genes at idx1 and idx2.
		
		If :data: is a tuple, :cell_size: should be a tuple of (inv_sf, inv_sf_sq). Otherwise, it should be an array of length data.shape[0].
	"""
	
	if type(data) == tuple:
		obs_M1 = (data[0]*data[2]*size_factor[0]).sum(axis=0)/n_obs
		obs_M2 = (data[1]*data[2]*size_factor[0]).sum(axis=0)/n_obs
		obs_MX = (data[0]*data[1]*data[2]*size_factor[1]).sum(axis=0)/n_obs
		cov = obs_MX - obs_M1*obs_M2

	else:

		idx1 = np.arange(0, data.shape[1]) if idx1 is None else np.array(idx1)
		idx2 = np.arange(0, data.shape[1]) if idx2 is None else np.array(idx2)
        
		overlap = set(idx1) & set(idx2)
		
		overlap_idx1 = [i for i in idx1 if i in overlap]
		overlap_idx2 = [i for i in idx2 if i in overlap]
		
		overlap_idx1 = [new_i for new_i,i in enumerate(idx1) if i in overlap ]
		overlap_idx2 = [new_i for new_i,i in enumerate(idx2) if i in overlap ]
		
		row_weight = (1/size_factor).reshape([1, -1]) if size_factor is not None else np.ones(data.shape[0]).reshape([1, -1])
		X, Y = data[:, idx1].T.multiply(row_weight).T.tocsr(), data[:, idx2].T.multiply(row_weight).T.tocsr()
		prod = (X.T*Y).toarray()/X.shape[0]
		prod[overlap_idx1, overlap_idx2] = prod[overlap_idx1, overlap_idx2] - data[:, overlap_idx1].T.multiply(row_weight**2).T.tocsr().sum(axis=0).A1/n_obs
		cov = prod - np.outer(X.mean(axis=0).A1, Y.mean(axis=0).A1)
					
	return cov/n_umi**2


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


def _corr_from_cov(cov, var_1, var_2, boot=False):
	"""
		Convert the estimation of the covariance to the estimation of correlation.
	"""
	
	if type(cov) != np.ndarray:
		return cov/np.sqrt(var_1*var_2)
	
	var_1[var_1 <= 0] = np.nan
	var_2[var_2 <= 0] = np.nan
	
	corr = np.full(cov.shape, np.nan)
	if boot:
		var_prod = np.sqrt(var_1)*np.sqrt(var_2)
	else:
		var_prod = np.outer(np.sqrt(var_1), np.sqrt(var_2))
	corr[np.isfinite(var_prod)] = cov[np.isfinite(var_prod)] / var_prod[np.isfinite(var_prod)]
	corr[~np.isfinite(corr)] = -np.inf
	
	corr[(corr < -1) | (corr > 1)] = np.nan
	
	return corr
