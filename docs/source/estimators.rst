Counts, normalization, and estimators
=====================================

The default estimator
---------------------

``setup_memento(..., estimator_type="hyper_relative")`` uses moment corrections
for an approximate hypergeometric RNA capture model. For observed count
:math:`x_i`, cell size factor :math:`s_i`, and a group capture rate :math:`q`,
the implementation estimates

.. math::

   \hat\mu = \frac{1}{n}\sum_i \frac{x_i}{s_i}, \qquad
   \hat v = \frac{1}{n}\sum_i
   \frac{x_i^2 - (1-q)x_i}{s_i^2} - \hat\mu^2.

The correction removes modeled capture noise from the second moment. An
estimated variance can be nonpositive in sparse or weakly varying data; this is
one reason moment estimation filters genes and hypothesis testing can omit
particular groups for a gene. Corrected covariance and the corresponding
variances define the gene-gene correlations used for 2D tests.

Capture rate
------------

Supply capture rates through an observation column named by ``q_column``.
Use finite fractions between 0 and 1. Setup uses the overall mean capture rate;
group moment calculations use each group's mean. This is a model input, not an
estimate of sequencing saturation from the count matrix. memento does not infer
an unknown capture rate automatically or calibrate different sequencing
chemistries for you. Assess plausible capture rates and the sensitivity of your
conclusions to that choice.

Size factors and residual variability
-------------------------------------

Setup first uses total counts to estimate a mean-variance trend, selects a
low-residual-variability set of genes, and estimates cell size factors from that
set with shrinkage. Use a broad gene panel for this stage. The defaults are
``trim_percent=0.1`` and ``shrinkage=0.5``.

After filtering, ``compute_1d_moments`` fits a quadratic trend between log mean
and log variance within each group. Residual variance is the exponential of the
log-variance residual from this trend. Differential variability tests use the
natural logarithm of that residual variance, allowing variability effects to be
examined after accounting for the expression-dependent trend.

Observed moments use the estimated size factors. Bootstrap moments use an
approximation that bins size factors (``num_bins=30`` by default). CPU and GPU
bootstrap paths retain this approximation. GPU acceleration does not remove the
size-factor correction.

Input scale and other estimators
--------------------------------

Use counts for both backends. Normalized or log-transformed values are not
interchangeable with counts, and converting those values to integers does not
recover the original measurements. Fractional count estimates from ambiguous
read assignment are not supported on GPU. The CPU sampler can retain fractional
values, but that does not model the additional uncertainty introduced by the
read-assignment algorithm.

The CPU implementation also contains alternative mean and moment estimators.
Their availability is not a statement that every estimator supports every
workflow. GPU 1D and 2D testing currently supports only ``hyper_relative``;
``mean_only`` and other estimators do not use the GPU path.
