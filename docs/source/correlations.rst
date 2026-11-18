Differential correlation
========================

Prepare counts, groups, 1D moments, and the design as in :doc:`basic`. Correlation
testing requires an explicit list of gene pairs whose genes survived filtering.
For a small illustrative panel:

.. code-block:: python

   from itertools import combinations

   panel = adata.var_names[:10].tolist()  # replace with your biological panel
   gene_pairs = list(combinations(panel, 2))
   memento.compute_2d_moments(adata, gene_pairs)
   memento.ht_2d_moments(
       adata,
       treatment=treatment,
       covariate=covariate,
       num_boot=10000,
       num_cpus=4,
       random_state=5,
       approx="norm",
       resample_rep=False,
       backend="cpu",  # or "gpu"
   )
   correlations = memento.get_2d_ht_result(adata)

Each row contains ``gene_1``, ``gene_2``, ``tx``, ``corr_coef``, ``corr_se``, and
``corr_pval``. The coefficient is the observed covariate-adjusted effect on
**raw correlation**. It is not a log fold change or a Fisher-z effect.
P-values are unadjusted; control multiple testing over your chosen pair-treatment
family. Self-pairs do not provide a differential correlation test and return NaN.

Both backends support ``resample_rep=True`` with the same group-row resampling
semantics described in :doc:`inference`. The GPU backend computes variances and
covariance from shared cell bootstrap weights, then performs regression on CUDA.
Its ``gpu_batch_size`` counts pairs rather than genes.

One-sample test
---------------

To ask whether each pair's correlation differs from zero, rather than whether
it differs between conditions, omit the treatment:

.. code-block:: python

   memento.ht_2d_moments(adata, num_boot=5000, random_state=5)
   correlations = memento.get_2d_ht_result(adata)  # tx == "intercept"

The coefficient is the cell-count-weighted mean of the group correlations, and
its bootstrap null is the same weighted mean over each bootstrap draw. Passing
a treatment whose columns are all constant is an error: constant columns are
removed as having nothing to test, so use ``treatment=None`` instead of an
all-ones column.

By default the standard error reflects only cell sampling within the groups
you have. To also account for variation between groups, such as donors, add
``resample_rep=True``. Group-level covariates (continuous or indicator
columns, one row per group) then remove the between-group variation they
explain before groups are resampled:

.. code-block:: python

   covariate = pd.DataFrame({"age": age, "batch_2": batch == 2}, index=groups, dtype=float)
   memento.ht_2d_moments(adata, covariate=covariate, resample_rep=True)

Covariates never change the coefficient. The correlation is estimated within
each group, so a group-level covariate cannot confound it, and adjusting the
weighted mean to the covariates' weighted average gives the weighted mean
back. For the same reason, a covariate without ``resample_rep=True`` leaves
the coefficient, standard error and p-value unchanged. To estimate the
correlation in a subpopulation, test only those groups; to ask whether the
correlation varies with a covariate, pass it as ``treatment``. When a donor
contributes several groups, add ``cluster="donor"`` to resample whole donors
(see :doc:`inference`). The one-sample form takes no per-pair dictionaries and
runs on either backend.

Groups with more cells carry more weight; inspect the per-group correlations
from ``get_2d_moments(adata, groupby=...)`` before relying on individual
pairs. The `one-sample tutorial
<https://github.com/yelabucsf/scrna-parameter-estimation/blob/master/tutorials/one_sample_correlations.ipynb>`_
walks through both tests and this check.

Select treatments or covariates per pair with tuple keys:

.. code-block:: python

   pair = gene_pairs[0]
   memento.ht_2d_moments(
       adata,
       treatment=treatment,
       covariate=covariate,
       treatment_for_gene={pair: ["stim"]},
       num_boot=10000,
       backend="gpu",
       random_state=5,
   )

Compute the requested pairs first; a misspelled or uncomputed pair raises an
error. For large panels, choose pairs based on the scientific question: testing
all combinations of thousands of genes creates millions of tests and a large
multiple-testing burden.
