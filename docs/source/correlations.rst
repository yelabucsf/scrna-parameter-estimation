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
