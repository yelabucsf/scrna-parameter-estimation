Bootstrap inference
===================

Within-group cell uncertainty
-----------------------------

For each retained group, memento resamples cells with replacement and recomputes
moments, using the binned size factors described in :doc:`estimators`. CPU
sampling compresses expression states per gene or pair. GPU sampling instead
forms multinomial cell weights shared across a batch of genes or pairs and uses
matrix multiplication. These preserve the per-test sampling distribution, but
CPU and GPU runs do not produce identical random draws or identical tables.

With ``resample_rep=False`` (the default), group membership and the design stay
fixed throughout regression. This measures uncertainty from sampling cells
within the observed groups, conditional on those groups and covariates.

Replicate resampling
--------------------

``resample_rep=True`` adds a second sampling stage on both CPU and GPU, for
``ht_1d_moments`` and ``ht_2d_moments``. For each gene or pair:

1. Retain valid groups and bootstrap moment columns. Regress the moments and
   treatment on the supplied covariates, with an intercept and weights equal
   to group cell counts.
2. For each replicate-bootstrap draw, sample as many group rows as there are
   valid groups, uniformly with replacement.
3. For each selected row, independently select one of its cell-bootstrap
   iterations. Resample its treatment residual and cell-count weight together.
4. Compute a weighted, centered slope from the selected residuals. Retain the
   original, unresampled coefficient separately as the observed statistic.

Covariate residualization happens once, before group resampling; the covariate
model is not refitted inside each replicate draw. Each gene or pair uses the
same group assignments for its moment types and assigned treatment columns.
Sampling is not stratified by treatment and does not keep multiple rows from
the same donor together. This implements the existing CPU group-row bootstrap;
it is not a paired-donor cluster bootstrap or a mixed-effects model.

Use this option when group rows correspond to the sampling units you intend
to resample. With very few groups, resampled treatments can be constant and
slopes undefined. Additional bootstrap draws cannot compensate for a lack of
independent biological replication. See :doc:`designs` for paired and unpaired
designs and :doc:`eqtl` for one-group-per-donor analysis.

Regression and reported statistics
----------------------------------

1D tests regress natural-log mean and natural-log residual variance. 2D tests
regress raw correlation. All regressions weight groups by cell count. Each
treatment column is tested separately after covariate adjustment; supplying
several treatment columns does not fit a joint treatment model.

For 1D tests, the reported coefficient is the mean of the valid coefficients,
including the observed coefficient and bootstrap coefficients. For 2D tests,
the reported coefficient is the observed coefficient. Standard errors are the
standard deviation of valid bootstrap coefficients, excluding the observed
coefficient (population standard deviation, ``ddof=0``).

With ``approx="norm"``, memento centers bootstrap coefficients by subtracting
the observed coefficient, fits a normal distribution to the finite values, and
computes a two-sided tail probability at the absolute observed effect. These
are the unadjusted ``*_pval`` columns. CPU also offers ``approx="boot"`` and
``approx="gdp"``; GPU currently supports only ``"norm"``.

Invalid moments and draws
-------------------------

Capture correction can produce invalid moments. Within a group, invalid
bootstrap moment values are replaced by samples from that group's valid
bootstrap values. A group without usable values is omitted for that gene or
pair. Correlation bootstrap values must lie strictly between -1 and 1.

A replicate draw with no treatment contrast has an undefined coefficient and
is excluded from finite bootstrap summaries. An empty usable null distribution,
an undefined observed effect, or a degenerate null scale gives a missing p-value.
Inspect missing results and retained groups; do not replace NaN p-values with
zero or interpret them as nonsignificance.

Reproducibility
---------------

Set ``random_state`` and record the software version, backend, bootstrap count,
and GPU batching/memory settings. CPU and GPU use different sampling algorithms.
Within a backend, a fixed seed and execution configuration are reproducible;
changing GPU batching or available memory can change the draws. GPU operations
use local generators and do not advance the global PyTorch random state.

Starting with this change, ``num_boot=B`` generates B replicate-bootstrap draws
plus the observed statistic. The previous CPU replicate-resampling path generated
B-1 draws because the observed statistic replaced one draw. Consequently, seeded
CPU results with ``resample_rep=True`` change even without selecting the GPU.
