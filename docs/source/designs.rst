Replicates, covariates, and comparisons
=======================================

Groups define which cells share a moment estimate. Treatments define the effects
you test. Covariates define what you adjust for. A variable can affect grouping
without automatically entering the regression.

Biological replicates and pairing
---------------------------------

For matched control and stimulated samples, the :doc:`basic` workflow groups by
``donor`` and ``stim`` and explicitly includes donor indicators as covariates.
With ``resample_rep=False``, uncertainty comes from resampling cells within the
observed groups, conditional on those groups. It does not estimate uncertainty
from sampling a new population of donors.

With ``resample_rep=True``, memento additionally samples group rows with
replacement. This can represent between-sample variation when each group is an
independent biological sample, for example one donor per row in an eQTL study.
It does not preserve pairs of rows belonging to the same donor, stratify by
condition, or implement random effects. Turning it on for donor-by-condition
rows does not create a paired cluster bootstrap. See :doc:`inference` for the
precise algorithm and its assumptions.

For an unpaired comparison with one group per donor, a donor indicator for every
donor would absorb the treatment effect. Use identifiable covariates such as
measured sample characteristics, and assess whether there are enough independent
samples for replicate resampling. More cells do not replace more donors.

What the binary wrappers do
---------------------------

``binary_test_1d`` and ``binary_test_2d`` are CPU convenience functions. Their
``replicates=["donor"]`` argument adds observation columns to the grouping:
``[treatment_col] + replicates``. It **does not add donor covariates**, enable
replicate resampling, or fit a paired model. Use the explicit setup, grouping,
moment, and testing functions when you need those controls or GPU execution.

``get_groups`` can return group metadata conveniently, but it converts numeric
labels to numbers and nonnumeric two-level labels to category codes. To control
the reference level and retain donor labels exactly, build metadata from
``adata.obs`` in retained group order as shown in :doc:`basic`.

Batch effects and continuous covariates
---------------------------------------

Include batch in the grouping when a desired sample stratum contains multiple
batches that should have separate moment estimates. Then encode batch as
regression covariates. If each donor belongs to exactly one batch, donor fixed
effects may already span that batch effect; avoid redundant columns.

For example, starting with a correctly ordered group metadata table:

.. code-block:: python

   # metadata has one row per retained group, with batch and age columns.
   covariate = pd.get_dummies(metadata[["batch"]], drop_first=True, dtype=float)
   covariate["age"] = metadata["age"].astype(float)

Continuous numeric covariates are supported in the linear group-level regression.
Categorical variables need numeric encoding. Nonlinear transformations and
interactions must be constructed explicitly. A cell-level covariate is not
included automatically: decide how it defines groups or a meaningful group-level
summary. Every design entry must be finite.

No covariate can resolve a treatment that is completely confounded with batch
or donor. Inspect the design before testing. A treatment with no variation is
omitted; a treatment explained by the covariates is not identifiable.

Several treatments versus one control
-------------------------------------

For comparisons such as drug A versus control and drug B versus control, run
separate binary analyses starting from the original counts object each time:
subset to the two relevant conditions, encode the selected drug as 1, and repeat
the :doc:`basic` workflow. Recreate groups and moments after subsetting.
Save each result table with its comparison label and choose a multiple-testing
family across comparisons if appropriate.

Multiple columns in ``treatment`` produce separate covariate-adjusted tests.
They are not a joint treatment model or an omnibus test; one treatment column
is not automatically adjusted for the others. For conditional effects, include
the desired adjustment variables explicitly in ``covariate`` and check that
the resulting design remains identifiable.

Tissue comparisons
------------------

The same procedure applies to fibroblasts or another cell type across tissues.
Subset to the cell type and two tissues, encode tissue as a binary treatment,
and group by donor and tissue. Use donor indicators for genuinely matched
samples. For unpaired tissues, choose covariates and resampling based on the
independent sample units rather than adding saturated donor indicators.
