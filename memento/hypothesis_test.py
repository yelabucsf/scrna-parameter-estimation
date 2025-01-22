"""
    hypothesis_test.py
    
    This file contains code to perform meta-analysis on the estimated parameters and their confidence intervals.
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
import scipy.sparse as sparse
from sklearn.linear_model import LinearRegression
import warnings
from functools import partial
from joblib import Parallel, delayed
import logging

import memento.util as util
import memento.bootstrap as bootstrap
import memento.estimator as estimator

def _robust_log(val):
    
    val[np.less_equal(val, 0., where=~np.isnan(val))] = np.nanmean(val)
    
    return np.log(val)


def _fill(val):
    
    condition = np.less_equal(val, 0., where=~np.isnan(val)) | np.isnan(val)
    num_invalid = condition.sum()
    
    if num_invalid == val.shape[0]:
        return None
    
    val[condition] = np.random.choice(val[~condition], num_invalid)
    
    return val

def _fill_corr(val):
    
    condition = np.less_equal(val, -1.0, where=~np.isnan(val)) | np.greater_equal(val, 1.0, where=~np.isnan(val)) | np.isnan(val)
        
    if val[~condition].shape[0] == 0:
        return None
    
    val[condition] = np.random.choice(val[~condition], condition.sum())
    
    return val


def _push(val, cond='neg'):
    
    if cond == 'neg':
        nan_idx = val < 0
    else:
        nan_idx = np.isnan(val)
    
    nan_count = nan_idx.sum()
    val[:(val.shape[0]-nan_count)] = val[~nan_idx]
    val[(val.shape[0]-nan_count):] = np.nan
    
    return val


def _compute_asl(perm_diff, approx='norm'):
    """ 
        Use the generalized pareto distribution to model the tail of the permutation distribution. 
    """

    if np.all(perm_diff == perm_diff.mean()):
        
        return np.nan
    
    null = perm_diff[1:] - perm_diff[0]
    
    null = null[np.isfinite(null)]
    
    stat = perm_diff[0]
    
    if approx == 'norm':
        
        dist = stats.norm
        null_params = dist.fit(null)
        
        abs_stat = np.abs(stat)
        
        return dist.sf(abs_stat, *null_params) + dist.cdf(-abs_stat, *null_params)

    if stat > 0:
        extreme_count = (null > stat).sum() + (null < -stat).sum()
    else:
        extreme_count = (null > -stat).sum() + (null < stat).sum()

    if extreme_count > 10 or approx == 'boot': # We do not need to use the GDP approximation. 

        return (extreme_count) / (null.shape[0])
        # return (extreme_count+1) / (null.shape[0]+1)

    if approx == 'gdp': # We use the GDP approximation, approx == 'gdp'
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:

                perm_dist = np.sort(null)# if perm_mean < 0 else np.sort(-perm_diff) # For fitting the GDP later on

                # Left tail
                N_exec = 300
                left_fit = False
                while N_exec > 50:

                    tail_data = perm_dist[:N_exec]
                    params = stats.genextreme.fit(tail_data)
                    _, ks_pval = stats.kstest(tail_data, 'genextreme', args=params)

                    if ks_pval > 0.05: # roughly a genpareto distribution
                        val = stats.genextreme.cdf(-np.abs(stat), *params)
                        left_asl = (N_exec/perm_dist.shape[0]) * val
                        left_fit = True
                        break
                    else: # Failed to fit genpareto
                        N_exec -= 30

                if not left_fit:
                    return (extreme_count+1) / (null.shape[0]+1)

                # Right tail
                N_exec = 300
                while N_exec > 50:

                    tail_data = perm_dist[-N_exec:]
                    params = stats.genextreme.fit(tail_data)
                    _, ks_pval = stats.kstest(tail_data, 'genextreme', args=params)

                    if ks_pval > 0.05: # roughly a genpareto distribution
                        val = stats.genextreme.sf(np.abs(stat), *params)
                        right_asl = (N_exec/perm_dist.shape[0]) * val
                        return right_asl + left_asl
                    else: # Failed to fit genpareto
                        N_exec -= 30                    

                return (extreme_count+1) / (null.shape[0]+1)

            except: # catch any numerical errors

                # Failed to fit genpareto, return the upper bound
                return (extreme_count+1) / (null.shape[0]+1)
    else:
        raise ValueError('approx must be one of ["boot", "norm", or "gdp"]!')
            

def _get_bootstrap_samples(
    true_mean, # list of means
    true_res_var, # list of residual variances
    cells, # list of sparse vectors/matrices
    approx_sf, # list of dense arrays
    covariate,
    treatment,
    Nc_list,
    num_boot,
    mv_fit, # list of tuples
    q, # list of numbers
    _estimator_1d,
    **kwargs):
    """
        Returns a tuple of ( <indices of good samples>, <bootstrap sample means>, <
    """
    
    good_idxs = np.zeros(treatment.shape[0], dtype=bool)
    
    # the resampled arrays
    boot_mean = np.zeros((treatment.shape[0], num_boot+1))*np.nan
    boot_var = np.zeros((treatment.shape[0], num_boot+1))*np.nan
    
    boot_means = []

    for group_idx in range(len(true_mean)):

        # Skip if any of the 1d moments are NaNs
        if np.isnan(true_mean[group_idx]) or \
           np.isnan(true_res_var[group_idx]) or \
           true_mean[group_idx] == 0 or \
           true_res_var[group_idx] < 0:
            continue

        # Fill in the true value
        boot_mean[group_idx, 0], boot_var[group_idx, 0] = true_mean[group_idx], true_res_var[group_idx]
                        
        # Generate the bootstrap values (us)
        mean, var = bootstrap._bootstrap_1d(
            data=cells[group_idx],
            size_factor=approx_sf[group_idx],
            num_boot=num_boot,
            q=q[group_idx],
            _estimator_1d=_estimator_1d,
            **kwargs)

        # Compute the residual variance
        res_var = estimator._residual_variance(mean, var, mv_fit[group_idx])

        # Minimize invalid values
        filled_mean = _fill(mean)#_push_nan(mean)#[:num_boot]
        filled_var = _fill(res_var)#_push_nan(res_var)#[:num_boot]

        # Make sure its a valid replicate
        if filled_mean is None or filled_var is None:
            continue

        boot_mean[group_idx, 1:] = filled_mean
        boot_var[group_idx, 1:] = filled_var
        
#         # This replicate is good
        good_idxs[group_idx] = True
        
    # Validity flag
    valid = (good_idxs.sum() != 0)
    
    return good_idxs, boot_mean, boot_var
    

def _ht_mean_quasiML(results, treatment, covariate, total_umi, umi_depth, group_names, gene_names, return_fits=False):
    """
    Differential mean expression using quasiML method, most suited for multi-sample data 
    where there are few samples but significant inter-sample variability. 
    
    :results: is a list of aggregate statistics for the mean from bootstrapping. 
    """
    
    results_df = pd.DataFrame(results)
    
    mean_df = pd.DataFrame([a[0] for a in results], columns=group_names, index=gene_names).T
    sem_df = pd.DataFrame([a[1] for a in results], columns=group_names, index=gene_names).T
    
    count_multiplier = total_umi#/umi_depth
    expr = mean_df*count_multiplier
    sampling_variance =  (sem_df**2)*count_multiplier**2
    
    # Fit loglinear models
    regressions = []
    for idx, gene in enumerate(expr.columns):

        # if treatment_for_gene is not None:
        #     if gene in treatment_for_gene: # Get treatments for this gene
        #         treatment_list = treatment_for_gene[gene]
        #     else: # Pass this gene
        #         continue
        # else: # Default, get all pairwise treatment-gene tests
        treatment_list = treatment.columns

        for t in treatment_list:

            design_matrix = pd.concat([covariate, treatment[[t]]], axis=1)

            regressions.append(
                partial(
                    util.fit_loglinear,
                    endog=expr.iloc[:, idx].values, 
                    exog=design_matrix,
                    offset=np.log(total_umi.flatten()), 
                    gene=gene, 
                    t=t))
    regression_fits = Parallel(n_jobs=14, verbose=1)(delayed(func)() for func in regressions)
    
    # For debugging purposes
    if return_fits:
        return regression_fits, sampling_variance
    
    # Clean up this code
    all_pred = np.array([fit['pred'] for fit in regression_fits]).T
    all_endog = np.array([fit['endog'] for fit in regression_fits]).T
    all_resid_variance = ((all_pred-all_endog)**2)
    
    gene_slopes = np.zeros(all_pred.shape[1])
    gene_intercepts = np.zeros(all_pred.shape[1])

    for idx in range(all_pred.shape[1]):

        e = all_endog[:, idx]
        m = all_pred[:, idx]
        v = all_resid_variance[:, idx]
        slope, inter, _, _, _ = stats.linregress(np.log10(m),np.log10(v))

        gene_slopes[idx] = slope
        gene_intercepts[idx] = inter
    
    mv_slope = np.clip(gene_slopes.mean(), a_min=0, a_max=2)
    alpha = 0.25 # TODO: pull as hyperparameter
    mv_slope = alpha*mv_slope + (1-alpha)*2
    logging.info(f'mv slope: {mv_slope}')
    
    result = []
    error_counter = 0
    for fit in regression_fits:
        coef = fit['model'].params[fit['t']]
        X = fit['design'].values
        pred = fit['pred']
        endog = fit['endog']

        intra_var = sampling_variance[fit['gene']].values
        inter_var = pred**mv_slope # from M-V relationship above
        total_var = intra_var + inter_var

        try:
            W = (pred**2) / total_var
            var = np.diag(np.linalg.pinv(X.T@np.diag(W)@X))[-1]
            se = np.sqrt(var)
            pv = 2*stats.norm.sf(np.abs(coef/se))
        except:
            se, pv = np.nan, np.nan
            error_counter+=1

        result.append((fit['gene'], fit['t'], coef, se, pv))

    return pd.DataFrame(result, columns=['gene', 'treatment', 'coef', 'se','pval']).set_index('gene')
    

def _ht_1d(
    true_mean, # list of means
    true_res_var, # list of residual variances
    cells, # list of sparse vectors/matrices
    approx_sf, # list of dense arrays
    covariate,
    treatment,
    Nc_list,
    num_boot,
    mv_fit, # list of tuples
    q, # list of numbers
    _estimator_1d,
    **kwargs):
    """
        Performs hypothesis testing of the mean and variance using bootstrap.
    """
    
    # Get the bootstrap sample statistics for each replicate
    good_idxs, boot_mean, boot_var = _get_bootstrap_samples(
        true_mean,
        true_res_var,
        cells,
        approx_sf,
        covariate,
        treatment,
        Nc_list,
        num_boot,
        mv_fit,
        q,
        _estimator_1d)
    
    vals = _regress_1d(
            covariate=covariate[good_idxs, :],
            treatment=treatment[good_idxs, :],
            boot_mean=np.log(boot_mean[good_idxs, :]), 
            boot_var=np.log(boot_var[good_idxs, :]),
            Nc_list=Nc_list[good_idxs],
            **kwargs)
    return vals


def _mean_summary_statistics(
    true_mean, # list of means
    true_res_var, # list of residual variances
    cells, # list of sparse vectors/matrices
    approx_sf, # list of dense arrays
    covariate,
    treatment,
    Nc_list,
    num_boot,
    mv_fit, # list of tuples
    q, # list of numbers
    _estimator_1d,
    **kwargs):
    """
        Performs hypothesis testing of the mean.
        
        Modes:
        - `bootstrap`: memento's approx bootstrap applied directly
        - `wls`: WLS regression based on bootstrap-estimated standard errors
        - `quasi-GLM`: Applies a global variance estimation and correction
    """
    # Get the bootstrap sample statistics for each replicate
    good_idxs, boot_mean, boot_var = _get_bootstrap_samples(
        true_mean,
        true_res_var,
        cells,
        approx_sf,
        covariate,
        treatment,
        Nc_list,
        num_boot,
        mv_fit,
        q,
        _estimator_1d)
    
    sem = np.nanstd(boot_mean, axis=1)
    pos_boot_mean = boot_mean.copy()
    pos_boot_mean[pos_boot_mean<0] = np.nan
    selm = np.nanstd(np.log(pos_boot_mean), axis=1)
    sel1pm = np.nanstd(np.log(pos_boot_mean+1), axis=1)
    
    return np.nanmean(boot_mean, axis=1), sem, selm, sel1pm
    return np.array(true_mean), sem, selm, sel1pm 


def _cross_coef(A, B, sample_weight):
    
    # Rowwise mean of input arrays & subtract from input arrays themeselves
    A_mA = A - np.average(A, axis=0, weights=sample_weight)
    B_mB = B - np.average(B, axis=0, weights=sample_weight)

    # Sum of squares across rows
    ssA = np.average(A_mA**2, axis=0, weights=sample_weight)

    # Finally get corr coeff
    return A_mA.T.dot(np.diag(sample_weight)).dot(B_mB)/sample_weight.sum() / ssA[:, None]


def _cross_coef_resampled(A, B, sample_weight):

    B_mB = B - np.average(B, axis=0, weights=sample_weight)
    A_mA = A - (A*sample_weight[:, :, np.newaxis]).sum(axis=0)/sample_weight.sum(axis=0)[:, np.newaxis]

    # Sum of squares across rows
    ssA = (A_mA**2*sample_weight[:, :, np.newaxis]).sum(axis=0)/sample_weight.sum(axis=0)[:, np.newaxis]
#     ssB = ((B_mB**2*sample_weight).sum(axis=0)/sample_weight.sum(axis=0))[:, np.newaxis]

    beta = np.einsum('ijk,ij->jk', A_mA * sample_weight[:, :, np.newaxis], B_mB).T/sample_weight.sum(axis=0) / ssA.T
#     r = beta * np.sqrt(ssA).T/np.sqrt(ssB).T
#     num_cells_per_boot = sample_weight.sum(axis=0)
#     t = np.sqrt(num_cells_per_boot-2)*r/np.sqrt(1-r**2)
#     se = beta/t
#     num_cell_correction = np.sqrt(num_cells_per_boot/num_cells_per_boot[0])
    return beta


def _regress_1d(covariate, treatment, boot_mean, boot_var, Nc_list, resample_rep=False,**kwargs):
    """
        Performs hypothesis testing for a single gene for many bootstrap iterations.
        
        Here, :X_center:, :X_center_Sq:, :boot_var:, :boot_mean: should have the same number of rows
    """
    
    valid_boostrap_iters = ~np.any(~np.isfinite(boot_mean), axis=0) & ~np.any(~np.isfinite(boot_var), axis=0)
    boot_mean = boot_mean[:, valid_boostrap_iters]
    boot_var = boot_var[:, valid_boostrap_iters]
    
    num_boot = boot_mean.shape[1]-1
    num_rep = boot_mean.shape[0]

    if boot_var.shape[1] == 0:

        print('skipped')

        return [np.zeros(treatment.shape[1])*np.nan]*5
    
    if (treatment == 1).mean()==1:
        
        mean_coef = np.average(boot_mean, axis=0, weights=Nc_list).reshape(1, -1)
        var_coef = np.average(boot_var, axis=0, weights=Nc_list).reshape(1, -1)
        
    else:

        boot_mean_tilde = boot_mean - LinearRegression(n_jobs=1).fit(covariate,boot_mean, Nc_list).predict(covariate)
        boot_var_tilde = boot_var - LinearRegression(n_jobs=1).fit(covariate,boot_var, Nc_list).predict(covariate)
        treatment_tilde = treatment - LinearRegression(n_jobs=1).fit(covariate, treatment, Nc_list).predict(covariate)

        if resample_rep:            

            replicate_assignment = np.random.choice(num_rep, size=(num_rep, num_boot))
            replicate_assignment[:, 0] = np.arange(num_rep)
            b_iter_assignment = np.random.choice(num_boot, (num_rep, num_boot))+1
            b_iter_assignment[:, 0] = 0

            boot_mean_resampled = boot_mean_tilde[(replicate_assignment, b_iter_assignment)]
            boot_var_resampled = boot_var_tilde[(replicate_assignment, b_iter_assignment)]
            treatment_resampled = treatment_tilde[replicate_assignment]
            weights_resampled = Nc_list[replicate_assignment]

            mean_coef = _cross_coef_resampled(treatment_resampled, boot_mean_resampled, weights_resampled)
            var_coef = _cross_coef_resampled(treatment_resampled, boot_var_resampled, weights_resampled)    

        else:

            mean_coef = _cross_coef(treatment_tilde, boot_mean_tilde, Nc_list)
            var_coef = _cross_coef(treatment_tilde, boot_var_tilde, Nc_list)

    mean_asl = np.apply_along_axis(lambda x: _compute_asl(x, **kwargs), 1, mean_coef)
    var_asl = np.apply_along_axis(lambda x: _compute_asl(x, **kwargs), 1, var_coef)

    mean_se = np.nanstd(mean_coef[:, 1:], axis=1)
    var_se = np.nanstd(var_coef[:, 1:], axis=1)
    
    mean_coef = np.nanmean(mean_coef, axis=1)
    var_coef = np.nanmean(var_coef, axis=1)

    return mean_coef, mean_se, mean_asl, var_coef, var_se, var_asl


def _ht_2d(
    true_corr, # list of correlations for each group
    cells, # list of Nx2 sparse matrices
    approx_sf,
    covariate,
    treatment,
    Nc_list,
    num_boot,
    q,
    _estimator_1d,
    _estimator_cov,
    **kwargs):
    
        
    good_idxs = np.zeros(treatment.shape[0], dtype=bool)
    
    # the bootstrap arrays
    boot_corr = np.zeros((treatment.shape[0], num_boot+1))*np.nan
    
    for group_idx in range(treatment.shape[0]):

        # Skip if any of the 2d moments are NaNs
        if np.isnan(true_corr[group_idx]) or (np.abs(true_corr[group_idx]) == 1):
            continue

        # Fill in the true value
        boot_corr[group_idx, 0] = true_corr[group_idx]
                
        # Generate the bootstrap values
        cov, var_1, var_2 = bootstrap._bootstrap_2d(
            data=cells[group_idx],
            size_factor=approx_sf[group_idx],
            num_boot=int(num_boot),
            q=q[group_idx],
            _estimator_1d=_estimator_1d,
            _estimator_cov=_estimator_cov,
            precomputed=None)

        corr = estimator._corr_from_cov(cov, var_1, var_2, boot=True)

        # This replicate is good
        vals = _fill_corr(corr)

        # Skip if all NaNs
        if vals is None:
            continue

        boot_corr[group_idx, 1:] = vals
        
        good_idxs[group_idx] = True


    # Skip this gene
    if good_idxs.sum() == 0:
        return np.nan, np.nan, np.nan
    
    vals = _regress_2d(
            covariate=covariate[good_idxs, :],
            treatment=treatment[good_idxs, :],
            boot_corr=boot_corr[good_idxs, :],
            Nc_list=Nc_list[good_idxs],
            **kwargs)
    
    return vals


def _regress_2d(covariate, treatment, boot_corr, Nc_list, resample_rep=False, **kwargs):
    """
        Performs hypothesis testing for a single pair of genes for many bootstrap iterations.
    """    
    
    valid_boostrap_iters = ~np.any(~np.isfinite(boot_corr), axis=0)
    boot_corr = boot_corr[:, valid_boostrap_iters]
    
    num_boot = boot_corr.shape[1]-1
    num_rep = boot_corr.shape[0]

    if boot_corr.shape[1] == 0:

        print('skipped')

        return [np.zeros(treatment.shape[1])*np.nan]*5
    
    if (treatment == 1).mean()==1:
        
        corr_coef = np.average(boot_corr, axis=0, weights=Nc_list).reshape(1,-1)
        
    else:

        boot_corr_tilde = boot_corr - LinearRegression(n_jobs=1).fit(covariate, boot_corr, Nc_list).predict(covariate)
        treatment_tilde = treatment - LinearRegression(n_jobs=1).fit(covariate, treatment, Nc_list).predict(covariate)

        if resample_rep:

            replicate_assignment = np.random.choice(num_rep, size=(num_rep, num_boot))
            replicate_assignment[:, 0] = np.arange(num_rep)
            b_iter_assignment = np.random.choice(num_boot, (num_rep, num_boot))+1
            b_iter_assignment[:, 0] = 0

            boot_corr_resampled = boot_corr_tilde[(replicate_assignment, b_iter_assignment)]
            treatment_resampled = treatment_tilde[replicate_assignment]
            weights_resampled = Nc_list[replicate_assignment]

            corr_coef = _cross_coef_resampled(treatment_resampled, boot_corr_resampled, weights_resampled)

        else:

            corr_coef = _cross_coef(treatment_tilde, boot_corr_tilde, Nc_list)

    corr_asl = np.apply_along_axis(lambda x: _compute_asl(x, **kwargs), 1, corr_coef)

    corr_se = np.nanstd(corr_coef[:, 1:], axis=1)

    return corr_coef[:, 0], corr_se, corr_asl