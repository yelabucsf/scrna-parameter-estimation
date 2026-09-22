Differential mean and variability
=================================

This workflow compares stimulated and control cells within one cell type, with
matched donors represented as fixed effects. Adapt the metadata column names
and capture rate to your experiment. See :doc:`designs` for unpaired samples,
batches, and multiple treatments.

Prepare counts and metadata
---------------------------

``adata.X`` must contain unnormalized, nonnegative counts in a SciPy CSR matrix.
Do not use log-transformed, scaled, or batch-corrected expression. Start from a
broad gene panel so that normalization and the mean-variance fit have enough
genes; select specific test genes after that preparation.

.. code-block:: python

   import numpy as np
   import pandas as pd
   import scanpy as sc
   from scipy import sparse
   import memento

   source = sc.read_h5ad("counts.h5ad")
   keep = (
       source.obs["cell_type"].eq("CD14+ Monocytes")
       & source.obs["condition"].isin(["ctrl", "stim"])
   )
   adata = source[keep].copy()
   # If counts are stored in a layer, use that instead:
   # adata.X = adata.layers["counts"].copy()
   adata.X = sparse.csr_matrix(adata.X)
   adata.obs["stim"] = adata.obs["condition"].eq("stim").astype(int)
   adata.obs["capture_rate"] = 0.07  # illustrative; choose for your experiment

   assert np.isfinite(adata.X.data).all() and (adata.X.data >= 0).all()
   assert adata.var_names.is_unique
   assert not adata.obs[["donor", "stim", "capture_rate"]].isna().any().any()

``capture_rate`` is the modeled fraction of RNA molecules captured, not the
sequencing saturation metric. See :doc:`estimators` for its role and limitations.
The GPU backend additionally requires integer-valued counts.

Estimate group moments
----------------------

.. code-block:: python

   memento.setup_memento(adata, q_column="capture_rate", min_cell_count=20)
   memento.create_groups(adata, label_columns=["donor", "stim"])
   memento.compute_1d_moments(adata, min_perc_group=0.7)

Each retained donor-condition combination is a group. Groups with fewer than
``min_cell_count`` cells are omitted. By default, ``compute_1d_moments`` also
subsets ``adata`` to genes with sufficient expression and positive estimated
variance in more than ``min_perc_group`` of groups. Inspect the retained groups
before interpreting a comparison; missing donor-condition combinations affect
the design. The threshold controls filtering, not statistical significance.

Construct the design in group order
-----------------------------------

Treatment and covariate DataFrames need one row per **retained group**, in exactly
``adata.uns["memento"]["groups"]`` order. Row labels alone do not reorder the
arrays for you. Reading original metadata avoids implicit category encoding:

.. code-block:: python

   group_order = adata.uns["memento"]["groups"]
   metadata = (
       adata.obs.drop_duplicates("memento_group")
       .set_index("memento_group")
       .loc[group_order, ["donor", "stim"]]
   )
   treatment = metadata[["stim"]].astype(float)
   covariate = pd.get_dummies(metadata["donor"], drop_first=True, dtype=float)
   if covariate.shape[1] == 0:
       covariate = pd.DataFrame({"intercept": 1.0}, index=metadata.index)

   assert treatment.index.tolist() == group_order
   assert covariate.index.tolist() == group_order

The regression includes an intercept internally. Donor indicators adjust for
donor-specific baselines. The ``stim`` coefficient is oriented as stimulated
minus control because the treatment was explicitly encoded as 1 versus 0.

Run the tests
-------------

.. code-block:: python

   memento.ht_1d_moments(
       adata,
       treatment=treatment,
       covariate=covariate,
       num_boot=10000,
       num_cpus=4,
       random_state=5,
       approx="norm",
       resample_rep=False,
   )
   result = memento.get_1d_ht_result(adata)

For an installed and working GPU backend, add ``backend="gpu"`` to this call.
``num_cpus`` controls CPU gene-level parallelism; it does not set GPU parallelism.
Both backends support ``resample_rep=True``, but that option resamples individual
group rows and is not a paired-donor bootstrap. Choose it based on the sampling
unit of your experiment, as explained in :doc:`inference`.

Read results and control multiple testing
-----------------------------------------

Each row is one gene-treatment test. The result contains:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Columns
     - Meaning
   * - ``gene``, ``tx``
     - Gene and tested treatment column.
   * - ``de_coef``, ``de_se``, ``de_pval``
     - Effect on natural-log mean expression, bootstrap standard error,
       and unadjusted p-value.
   * - ``dv_coef``, ``dv_se``, ``dv_pval``
     - Effect on natural-log residual variance, bootstrap standard error,
       and unadjusted p-value.

For a binary treatment, ``de_coef / np.log(2)`` expresses the mean effect on a
log2 scale. ``dv_coef`` describes variability beyond the fitted mean-variance
trend; it is not the change in raw variance. See :doc:`inference` for the exact
coefficient summaries and p-value construction.

memento does not automatically adjust these p-values for multiple testing.
For example, apply Benjamini-Hochberg separately to the mean and variability
families, excluding missing p-values:

.. code-block:: python

   from statsmodels.stats.multitest import multipletests

   for prefix in ("de", "dv"):
       finite = np.isfinite(result[f"{prefix}_pval"])
       result[f"{prefix}_fdr"] = np.nan
       if finite.any():
           result.loc[finite, f"{prefix}_fdr"] = multipletests(
               result.loc[finite, f"{prefix}_pval"], method="fdr_bh"
           )[1]

Define the testing family across treatments or cell types to match the scientific
claims you intend to make. A missing result is not evidence of no effect.

Analysis state
--------------

Preparation functions and tests normally update ``adata`` in place; filtering
can remove genes. Keep an original counts object or work on a copy. Test results
live under ``adata.uns["memento"]`` and repeated tests overwrite the corresponding
result slot. Save returned result DataFrames before running another comparison.
``ht_1d_moments(..., inplace=False)`` returns an analyzed copy of AnnData; retrieve
its table with ``get_1d_ht_result``.
