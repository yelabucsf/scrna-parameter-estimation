Release notes
=============

0.1.4
-----

One-sample correlation tests
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **Changed usage.** To test correlations against zero, call
  ``ht_2d_moments(adata, treatment=None)``. In 0.1.3 this was requested with a
  treatment column equal to 1 for every group (for example ``Inter = 1``);
  that form now raises an error pointing to ``treatment=None``. The estimate
  is unchanged: the cell-count-weighted mean of the group correlations.
* The one-sample test rejects covariates, per-pair dictionaries and
  ``resample_rep`` instead of silently ignoring them as 0.1.3 did. Covariates
  cannot change a one-sample average; select the population by subsetting.
* ``tutorials/one_sample_correlations.ipynb`` is rewritten for this API and
  adds a per-donor consistency check.

Replicate resampling
~~~~~~~~~~~~~~~~~~~~

* New ``cluster`` option for ``ht_1d_moments`` and ``ht_2d_moments`` (CPU and
  GPU): with ``resample_rep=True``, ``cluster="donor"`` resamples whole donors and
  keeps all of a donor's group rows together. Use it when a donor contributes
  several groups, such as donor-by-batch or donor-by-condition designs, where
  resampling rows independently understates standard errors.
* Fixed: with ``resample_rep=True``, a draw whose treatment values differed
  only by rounding error could produce a slope around 1e14 and distort the
  reported 1D coefficient. Such draws are now treated as having no contrast on
  both the CPU and GPU backends.
* CPU replicate resampling now produces exactly ``num_boot`` draws in addition
  to the observed statistic. This fixes an off-by-one error and changes seeded
  results when ``resample_rep=True``. Empty usable null distributions return
  NaN. Invalid observed statistics are not replaced by bootstrap draws during
  finite-value filtering.

Designs and inputs
~~~~~~~~~~~~~~~~~~

* Treatment columns that are constant across groups (for example a monomorphic
  SNP) are dropped with a warning before testing. A global design whose columns
  are all constant raises an error.
* Fixed: ``ht_2d_moments`` on the CPU backend ignored ``covariate_for_gene``
  and applied every covariate column to every pair.
* ``ht_2d_moments`` raises an error for gene pairs that were not passed to
  ``compute_2d_moments``, instead of silently skipping them.
* Undefined group correlations are reported as NaN by ``get_2d_moments``.
  Previously they appeared as exactly +1 or -1. Test results are unchanged.
* Deprecated: the ``replicates`` argument of ``binary_test_1d`` and
  ``binary_test_2d`` emits a ``FutureWarning`` and will be removed. It only
  adds columns to the grouping and does not adjust for replicates.
* Compatible with NumPy 2.

GPU backend
~~~~~~~~~~~

* Optional GPU bootstrap and regression for differential mean, variability, and
  correlation, using the existing analysis functions with ``backend="gpu"``.
  Default installation and CPU execution do not require PyTorch.
* Adaptive GPU memory planning, cell-weight caching/streaming, and gene- or
  pair-specific treatment/covariate dictionaries.
* GPU replicate resampling for ``ht_1d_moments`` and ``ht_2d_moments``, matching
  the CPU group-row bootstrap and handling undefined resampled slopes.
* Expanded installation, design, inference, correlation, and eQTL documentation.

0.1.3
-----

* Adaptive bootstrap sampling for differential mean testing (``ht_mean``) and
  faster bootstrap generation.
* Nonnumeric two-level treatment labels are encoded automatically by
  ``get_groups``.
* Fixed a division by zero for constant design-matrix columns.
* Empty groups raise an informative error when fitting the mean-variance trend.
