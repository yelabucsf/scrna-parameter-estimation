Release notes
=============

0.1.3 development
-----------------

* Optional GPU bootstrap and regression for differential mean, variability, and
  correlation, using the existing analysis functions with ``backend="gpu"``.
  Default installation and CPU execution do not require PyTorch.
* Adaptive GPU memory planning, cell-weight caching/streaming, and gene- or
  pair-specific treatment/covariate dictionaries.
* GPU replicate resampling for ``ht_1d_moments`` and ``ht_2d_moments``, matching
  the CPU group-row bootstrap and handling undefined resampled slopes.
* CPU replicate resampling now produces exactly ``num_boot`` draws in addition
  to the observed statistic. This fixes an off-by-one error and changes seeded
  results when ``resample_rep=True``. Empty usable null distributions return
  NaN; constant resampled treatment columns cannot produce a finite slope from
  rounding error. Invalid observed statistics are not replaced by bootstrap
  draws during finite-value filtering.
* Expanded installation, design, inference, correlation, and eQTL documentation,
  including the grouping-only meaning of the binary wrappers' ``replicates``
  argument.
