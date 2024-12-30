"""
	mean.py
	
	Contains the estimators and hypothesis testing framework for the mean.
	
"""

import scipy.sparse as sparse
import scipy.stats as stats
import scipy.interpolate as inter
import numpy as np
import pickle as pkl


def _pseudobulk(data: sparse.csc_matrix, size_factor: np.array)
    """ 
        Computes the pseudobulk mean with pseudocount adjustment. 
        :X: is a sparse matrix in CSC format. 
    """"
                
                
    mean = (data.sum(axis=0).A1+1)/(size_factor.sum()+1)
    
    return m, np.ones(m.shape)*10
    
def _hyper_1d_relative(data, n_obs, q, size_factor=None):
    """
        Estimate the variance using the hypergeometric noise process.
        
        If :data: is a tuple, :cell_size: should be a tuple of (inv_sf, inv_sf_sq). Otherwise, it should be an array of length data.shape[0].
    """
    if type(data) == tuple:
        size_factor = size_factor if size_factor is not None else (1, 1)
        mm_M1 = (data[0]*data[1]*size_factor[0]).sum(axis=0)/n_obs
        mm_M2 = (data[0]**2*data[1]*size_factor[1] - (1-q)*data[0]*data[1]*size_factor[1]).sum(axis=0)/n_obs
    else:
        
        row_weight = (1/size_factor).reshape([1, -1])
        row_weight_sq = (1/size_factor**2).reshape([1, -1])
        mm_M1 = sparse.csc_matrix.dot(row_weight, data).ravel()/n_obs
        mm_M2 = sparse.csc_matrix.dot(row_weight_sq, data.power(2)).ravel()/n_obs - (1-q)*sparse.csc_matrix.dot(row_weight_sq, data).ravel()/n_obs
    
    mm_mean = mm_M1
    mm_var = (mm_M2 - mm_M1**2)

    return [mm_mean, mm_var]


def _mean_only_1p(data, n_obs, q, size_factor=None):
    """
        Estimate the variance using the Poisson noise process.
        
        If :data: is a tuple, :cell_size: should be a tuple of (inv_sf, inv_sf_sq). Otherwise, it should be an array of length data.shape[0].
    """
    if type(data) == tuple:
        size_factor = size_factor if size_factor is not None else (1, 1)
        mm_M1 = (data[0]*data[1]*size_factor[0]).sum(axis=0)/n_obs
    else:
        
        row_weight = (1/size_factor).reshape([1, -1])
        mm_M1 = sparse.csc_matrix.dot(row_weight, data).ravel()/n_obs
    
    mm_mean = mm_M1

    return [mm_mean+1, np.ones(mm_mean.shape)*10]


def _good_mean_only(data, n_obs, q, size_factor=None, alpha=0, max_to_replace=13):
    """
        Hypergeometric mean estimator based on Good's estimator.
    """

    if type(data) == tuple:
        return
    else:

        arr = data
        n_genes = data.shape[1]
        pb = (arr.sum(axis=0).A1+1)
        total_umi = pb.sum()
        denom = np.array([total_umi - pb[sparse.find(arr[i])[1]].sum() for i in range(n_obs)]).mean()

        freqs = bincount2d_sparse(arr)
        expected_freqs = freqs.mean(axis=0).A1
        initial_values = np.tile(np.arange(max_to_replace)[:,np.newaxis], (1,n_genes))
        final_values = (initial_values + 1) * expected_freqs[initial_values+1] / expected_freqs[initial_values]
        final_values[0] = 0#(0+1) * expected_freqs[1] * (alpha*(pb/denom) + ((1-alpha)/expected_freqs[0]))

        corrected_counts = sparse.csr_matrix(arr, dtype=float)
        for val in range(1,max_to_replace):
            corrected_counts.data[corrected_counts.data == val] = final_values[val, 0]

        corrected_counts = sparse.diags(1/size_factor) @ corrected_counts # normalize for size_factor
        nonzero_sum = corrected_counts.sum(axis=0).A1
        num_zeros_per_column = arr.shape[0] - arr.getnnz(axis=0)
        zero_sum = np.array([(final_values[0, idx]/size_factor[~np.in1d(range(size_factor.shape[0]), sparse.find(arr[:, idx])[0])]).sum() for idx in range(n_genes)])
        m = (nonzero_sum + zero_sum)/arr.shape[0]
        
        return [m, np.ones(m.shape)*10]




