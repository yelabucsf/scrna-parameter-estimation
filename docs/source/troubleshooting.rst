Troubleshooting and common questions
====================================

No genes with positive mean and variance
----------------------------------------

The mean-variance trend needs genes with positive corrected moments. Check raw
count input, per-group cell counts, expression sparsity, and the capture rate.
A group can pass the minimum cell count but still lack usable genes. Exclude or
pool groups only when justified by the design; review ``min_cell_count`` and
``filter_mean_thresh`` before relaxing filters. Start from a broad gene panel.

A gene or pair is missing
-------------------------

``compute_1d_moments`` filters genes by default. Build dictionaries and pair
lists from the retained ``adata.var_names``. ``compute_2d_moments`` must precede
2D testing. Constant treatment columns are omitted because there is no effect
to test. Empty dictionaries intentionally request no tests.

NaN statistics
--------------

Check valid groups, treatment variation after covariate adjustment, and whether
bootstrap draws have usable moments. A genotype main effect cannot be identified
with a separate indicator for every donor. Replicate resampling of very small
designs can frequently select rows with no treatment contrast. See :doc:`inference`.

Unexpected treatment direction or sample alignment
--------------------------------------------------

Encode the reference condition explicitly as 0 and the tested condition as 1.
Construct treatment and covariate rows in retained group order; the index does
not automatically align a DataFrame. ``get_groups`` can encode nonnumeric
binary labels, so inspect its output or use the metadata construction in
:doc:`basic`.

CUDA unavailable or out of memory
---------------------------------

Check ``torch.cuda.is_available()`` in the same environment used for analysis.
The pip extra does not install a driver or select the right CUDA wheel for every
machine. Verify the PyTorch installation first. For memory pressure, close
unneeded GPU workloads or lower the memory target/batch cap described in
:doc:`gpu`. Host RAM and GPU RAM are separate resources; loading large AnnData
layers can exhaust host memory before testing starts.

Different CPU and GPU p-values
------------------------------

The backends draw different bootstrap samples and use different arithmetic for
moments. Compare effects and standard errors as well as p-values, especially
near a significance threshold. Increase ``num_boot`` or repeat seeds to assess
Monte Carlo sensitivity. Exact equality is expected in fixed-input regression
tests within numerical tolerance, not between independently sampled analyses.

Older notebooks and missing classes
-----------------------------------

The current API uses functions such as ``setup_memento`` and ``ht_1d_moments``.
Some paper reproductions use a separate object-oriented implementation containing
``RNAHypergeometric``. Follow the repository's
`publication instructions <https://github.com/yelabucsf/scrna-parameter-estimation/blob/master/publication/figure2/README.md>`_
and its ``MEMENTO_OO_PATH`` configuration for those analyses. Historical notebooks
under ``publication/original`` may use obsolete imports such as ``memento.model``;
they are archival, not examples for the current package API.

Reporting a problem
-------------------

Include the memento version or Git commit, Python and relevant dependency versions,
backend, estimator, bootstrap options, shapes, group cell counts, and a small
reproducer. For GPU errors, include the PyTorch/CUDA versions, GPU model, available
memory, and memory settings. Synthetic counts and metadata are usually sufficient
to demonstrate a software problem without sharing the original dataset.
