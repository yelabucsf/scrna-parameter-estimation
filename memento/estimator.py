"""
    estimator.py
    
    Contains the estimators for the mean, variance, covariance, and the correlation. 
    
    Implements both the approximate hypergeometric and Poisson generative models. 
    
    If :data: is a tuple, it is assumed that it is a tuple of (value, count), both of which are 1D numpy arrays.
    If :data: is not a tuple, it is assumed to be a SciPy CSC sparse matrix.
"""

import scipy.sparse as sparse
import scipy.stats as stats
import scipy.interpolate as inter
import numpy as np
import pickle as pkl


def _get_estimator_1d(estimator_type):
    
    if estimator_type == 'hyper_absolute':
        return _hyper_1d_absolute
    elif estimator_type == 'hyper_relative':
        return _hyper_1d_relative
    elif estimator_type == 'poi_absolute':
        return _poi_1d_absolute
    elif estimator_type == 'poi_relative':
        return _poi_1d_relative
    elif estimator_type == 'mean_only':
        return _mean_only_1p
    elif estimator_type == 'good_mean_only':
        return _good_mean_only
    elif estimator_type == 'pseudobulk':
        return _pseudobulk
    else: # Custom 1D estimator
        return estimator_type[0]

    
def _get_estimator_cov(estimator_type):
    
    if estimator_type == 'hyper_absolute':
        return _hyper_cov_absolute
    elif estimator_type == 'hyper_relative':
        return _hyper_cov_relative
    elif estimator_type == 'poi_absolute':
        return _poi_cov_absolute
    elif estimator_type == 'poi_relative':
        return _poi_cov_relative
    else: # Custom covariance estimator
        return estimator_type[1]
    
    
def bincount2d_sparse(sparse_arr, bins=None):

    bins = np.round(sparse_arr.max()).astype(int) + 1
    num_cells, num_genes = sparse_arr.shape
    count = sparse.lil_matrix((num_cells, bins))
    for cell in range(num_cells):
        cell_counts = np.bincount(np.round(sparse_arr[cell].data).astype(int))
        cell_counts[0] = num_genes - sparse_arr[cell].nnz
        count[cell, np.arange(cell_counts.shape[0])] = cell_counts
    return count
    

def _estimate_size_factor(data, estimator_type, shrinkage, mask=None, total=False):
    """Calculate the size factor
    
    Args: 
        data (AnnData): the scRNA-Seq CG (cell-gene) matrix.
        
    Returns:
        size_factor((Nc,) ndarray): the cell size factors.
    """
    
    if 'absolute' in estimator_type:
        return np.ones(data.shape[0])
    
    X=data
    
    if total:
        Nrc = np.array(X.sum(axis=1)).reshape(-1)
        Nr = Nrc.mean()
        n_umi = np.array(X.sum(axis=1)).reshape(-1).mean()

        size_factor = Nrc
        
        return size_factor
        
    if mask is not None:
    
        Nrc = X.multiply(mask).sum(axis=1).A1.astype(float)
        if shrinkage > 0:
            Nrc += np.quantile(Nrc, shrinkage) # Shrinkage
        Nr = np.median(Nrc)
        size_factor = Nrc/Nr
        
        n_umi = np.median(np.array(X.sum(axis=1)).reshape(-1))
        size_factor = size_factor*n_umi

        return size_factor

    
def _fit_mv_regressor(mean, var):
    """
        Perform regression of the variance against the mean.
    """
    
    cond = (mean > 0) & (var > 0)
    m, v = np.log(mean[cond]), np.log(var[cond])
    
    poly = np.polyfit(m, v, 2)
    return poly
    f = np.poly1d(z)
    
#     spline = inter.UnivariateSpline(m[np.argsort(m)], v[np.argsort(m)])
#     return spline

#     slope, inter, _, _, _ = stats.linregress(m, v)
#     return slope, inter


def _residual_variance(mean, var, mv_fit):
    
    cond = (mean > 0) & (var > 0)
    rv = np.zeros(mean.shape)*np.nan
    
    f = np.poly1d(mv_fit)
    with np.errstate(invalid='ignore'):
        rv[cond] = np.exp(np.log(var[cond]) - f(np.log(mean[cond])))
    return rv


def _poisson_1d_relative(data, n_obs, size_factor=None):
    """
        Estimate the variance using the Poisson noise process.
        
        If :data: is a tuple, :cell_size: should be a tuple of (inv_sf, inv_sf_sq). Otherwise, it should be an array of length data.shape[0].
    """
    if type(data) == tuple:
        size_factor = size_factor if size_factor is not None else (1, 1)
        mm_M1 = (data[0]*data[1]*size_factor[0]).sum(axis=0)/n_obs
        mm_M2 = (data[0]**2*data[1]*size_factor[1] - data[0]*data[1]*size_factor[1]).sum(axis=0)/n_obs
    else:
        
        row_weight = (1/size_factor).reshape([1, -1]) if size_factor is not None else np.ones(data.shape[0])
        mm_M1 = sparse.csc_matrix.dot(row_weight, data).ravel()/n_obs
        mm_M2 = sparse.csc_matrix.dot(row_weight**2, data.power(2)).ravel()/n_obs - sparse.csc_matrix.dot(row_weight**2, data).ravel()/n_obs
    
    mm_mean = mm_M1
    mm_var = (mm_M2 - mm_M1**2)

    return [mm_mean, mm_var]


def _poisson_cov_relative(data, n_obs, size_factor, idx1, idx2):
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
        
        overlap_location = (idx1 == idx2)
        overlap_idx = [i for i,j in zip(idx1, idx2) if i == j]
        
        row_weight = np.sqrt(1/(size_factor**2)).reshape([1, -1])
        X, Y = data[:, idx1].T.multiply(row_weight).T.tocsr(), data[:, idx2].T.multiply(row_weight).T.tocsr()
        
        prod = X.multiply(Y).sum(axis=0).A1/n_obs
        if len(overlap_idx) >0:
            prod[overlap_location] = prod[overlap_location] - data[:, overlap_idx].T.multiply(row_weight**2).T.tocsr().sum(axis=0).A1/n_obs
        cov = prod - X.mean(axis=0).A1*Y.mean(axis=0).A1
                    
    return cov


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


def _pseudobulk(data: sparse.csc_matrix, n_obs, q, size_factor: np.array):
    """ 
        Computes the pseudobulk mean with pseudocount adjustment. 
        
        If :data: is a tuple, :cell_size: should be a tuple of (inv_sf, inv_sf_sq). Otherwise, it should be an array of length data.shape[0].
    """
                
    if type(data) == tuple:
        size_factor = size_factor if size_factor is not None else (1, 1)
        unique_expr, bootstrap_freq = data[0], data[1]
        inverse_size_factor, inverse_size_factor_sq = size_factor[0], size_factor[1]
        
        m = ((unique_expr * bootstrap_freq).sum(axis=0)+1) / ((bootstrap_freq/inverse_size_factor).sum(axis=0))
    else:
        m = (data.sum(axis=0).A1+1)/(size_factor.sum()+1)
        
    return m, np.ones(m.shape)*10


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


def _hyper_cov_relative(data, n_obs, size_factor, q, idx1=None, idx2=None):
    """
        Estimate the covariance using the hypergeometric noise process between genes at idx1 and idx2.
        
        If :data: is a tuple, :cell_size: should be a tuple of (inv_sf, inv_sf_sq). Otherwise, it should be an array of length data.shape[0].
    """
    
    if type(data) == tuple:
        obs_M1 = (data[0]*data[2]*size_factor[0]).sum(axis=0)/n_obs
        obs_M2 = (data[1]*data[2]*size_factor[0]).sum(axis=0)/n_obs
        obs_MX = (data[0]*data[1]*data[2]*size_factor[1]).sum(axis=0)/n_obs
        cov = obs_MX - obs_M1*obs_M2

    else:

        overlap_location = (idx1 == idx2)
        overlap_idx = [i for i,j in zip(idx1, idx2) if i == j]
        
        row_weight = np.sqrt(1/size_factor**2).reshape([1, -1])
        X, Y = data[:, idx1].T.multiply(row_weight).T.tocsr(), data[:, idx2].T.multiply(row_weight).T.tocsr()
        
        prod = X.multiply(Y).sum(axis=0).A1/n_obs
        if len(overlap_idx) >0:
            prod[overlap_location] = prod[overlap_location] - (1-q)*data[:, overlap_idx].T.multiply(row_weight**2).T.tocsr().sum(axis=0).A1/n_obs
        cov = prod - X.mean(axis=0).A1*Y.mean(axis=0).A1
                    
    return cov


def _hyper_corr_symmetric(data, n_obs, size_factor, q, var, idx1=None, idx2=None):
    """
        Estimate the correlation matrix given a dataset.
        This function is useful for computing an all by all correlation matrix for clustering, input to other algorithms.
    """

    idx1 = np.arange(0, data.shape[1])
    idx2 = np.arange(0, data.shape[1])

    overlap = set(idx1) & set(idx2)

    overlap_idx1 = [i for i in idx1 if i in overlap]
    overlap_idx2 = [i for i in idx2 if i in overlap]

    overlap_idx1 = [new_i for new_i,i in enumerate(idx1) if i in overlap]
    overlap_idx2 = [new_i for new_i,i in enumerate(idx2) if i in overlap]

    row_weight = np.sqrt(1/(size_factor**2)).reshape([1, -1])
    X, Y = data[:, idx1].T.multiply(row_weight).T.tocsr(), data[:, idx2].T.multiply(row_weight).T.tocsr()
    prod = (X.T*Y).toarray()/X.shape[0]
    prod[overlap_idx1, overlap_idx2] = prod[overlap_idx1, overlap_idx2] - (1-q)*data[:, overlap_idx1].T.multiply(row_weight**2).T.tocsr().sum(axis=0).A1/n_obs
    cov = prod - np.outer(X.mean(axis=0).A1, Y.mean(axis=0).A1)
    
    var_1 = var[idx1]
    var_2 = var[idx2]
    var_1[var_1 <= 0] = np.nan
    var_2[var_2 <= 0] = np.nan
    var_prod = np.sqrt(np.outer(var[idx1], var[idx2]))
    
    corr = np.full(cov.shape, 5.0)
    corr[np.isfinite(var_prod)] = cov[np.isfinite(var_prod)] / var_prod[np.isfinite(var_prod)]
    corr[(corr < 1.05) & (corr > -1.05)] = np.clip(corr[(corr < 1.05) & (corr > -1.05)], a_min=-1, a_max=1)
    corr[(corr > 1) | (corr < -1)] = np.nan
    
    return corr
    
    
def _corr_from_cov(cov, var_1, var_2, boot=False):
    """
        Convert the estimation of the covariance to the estimation of correlation.
    """
    
    if type(cov) != np.ndarray:
        return cov/np.sqrt(var_1*var_2)
        
    corr = np.full(cov.shape, 5.0)
        
    var_1[var_1 <= 0] = np.nan
    var_2[var_2 <= 0] = np.nan
    var_prod = np.sqrt(var_1*var_2)
        
    corr[np.isfinite(var_prod)] = cov[np.isfinite(var_prod)] / var_prod[np.isfinite(var_prod)]
    
    if not boot:
        corr[corr > 1] = 1
        corr[corr < -1] = -1
    
    return corr
