eQTLs, vQTLs, and cQTLs
=======================

Use genotype dosage as the treatment to test associations with mean expression
(eQTLs), residual variability (vQTLs), or gene-gene correlation (cQTLs). The same
donor-level design supports all three analyses on CPU or GPU.

Prepare the donor-level design
------------------------------

For a single cell type, group cells by donor so each retained group represents
one independent sample. Complete setup and 1D moment estimation before building
the design. This example assumes ``genotypes_by_donor`` contains numeric SNP
dosages with donor IDs as its index, and ``covariates_by_donor`` contains numeric
covariates such as genotype principal components, age, and encoded batch.

.. code-block:: python

   group_order = adata.uns["memento"]["groups"]
   donors = (
       adata.obs.drop_duplicates("memento_group")
       .set_index("memento_group").loc[group_order, "donor"]
   )
   genotypes = genotypes_by_donor.loc[donors.tolist()].copy()
   covariates = covariates_by_donor.loc[donors.tolist()].copy()
   genotypes.index = group_order
   covariates.index = group_order
   # Keys must name genes retained after compute_1d_moments.
   gene_to_snps = {"GENE_A": ["rs1", "rs2"], "GENE_B": ["rs3"]}

Use consistent allele coding, for example 0, 1, or 2 copies of the designated
effect allele. The coefficients then describe the change per additional copy
of that allele. Dosages and covariates must be finite numeric values; resolve
missing genotypes before constructing the design. With one group per donor,
``resample_rep=True`` resamples donors as described in :doc:`inference`.

eQTL and vQTL tests
-------------------

``ht_1d_moments`` tests mean and residual variability together. A single call
provides both eQTL and vQTL results for each requested gene-SNP association:

.. code-block:: python

   memento.ht_1d_moments(
       adata,
       treatment=genotypes,
       covariate=covariates,
       treatment_for_gene=gene_to_snps,
       resample_rep=True,
       approx="norm",
       num_boot=10000,
       random_state=5,
       backend="gpu",  # use "cpu" without the optional dependency
   )
   result = memento.get_1d_ht_result(adata)
   eqtls = result[["gene", "tx", "de_coef", "de_se", "de_pval"]].copy()
   vqtls = result[["gene", "tx", "dv_coef", "dv_se", "dv_pval"]].copy()

``tx`` identifies the SNP column. ``de_coef`` is the effect on natural-log mean
expression; divide by ``log(2)`` to express it on a log2 scale. ``dv_coef`` is the
effect on natural-log residual variance, after removing the fitted mean-variance
trend within each donor group. A positive vQTL coefficient means greater residual
variability with increasing effect-allele dosage. It does not measure variation
of donor means or an effect on unadjusted raw variance.

The ``*_se`` columns are bootstrap standard errors and ``*_pval`` columns are
unadjusted p-values. Both moment types use the same gene-to-SNP dictionary. A
vQTL test does not require the association to be significant as an eQTL first.
See :doc:`estimators` for residual variance and :doc:`inference` for coefficient
summaries.

cQTL tests
----------

Reuse the donor-ordered ``genotypes`` and ``covariates`` above. Map each gene-pair
tuple to its candidate SNP columns, then compute moments for those pairs before
testing. Both genes must have survived 1D moment filtering; replace the example
names with retained genes and genotype columns from your data.

.. code-block:: python

   pair_to_snps = {
       ("GENE_A", "GENE_B"): ["rs1", "rs2"],
       ("GENE_A", "GENE_C"): ["rs3"],
   }
   memento.compute_2d_moments(adata, gene_pairs=list(pair_to_snps))
   memento.ht_2d_moments(
       adata,
       treatment=genotypes,
       covariate=covariates,
       treatment_for_gene=pair_to_snps,
       resample_rep=True,
       approx="norm",
       num_boot=10000,
       random_state=5,
       backend="gpu",  # use "cpu" without the optional dependency
   )
   cqtls = memento.get_2d_ht_result(adata)

Although the argument is named ``treatment_for_gene``, its keys for 2D tests are
``(gene_1, gene_2)`` tuples matching the computed pairs. ``covariate_for_gene``
can likewise select covariate columns per pair.

The output contains ``gene_1``, ``gene_2``, ``tx``, ``corr_coef``, ``corr_se``, and
``corr_pval``. ``corr_coef`` is the observed covariate-adjusted change in **raw
correlation per effect-allele copy**. A positive coefficient indicates a more
positive correlation as dosage increases, which can include a negative
correlation becoming less negative. This coefficient is not a log fold change
or a Fisher-z effect. Avoid self-pairs and select candidate pairs and variants
to match the biological question; all-by-all panels produce many tests.

Multiple testing
----------------

Choose the family of associations for which you want to control false discovery.
For example, to apply Benjamini-Hochberg separately across all tested eQTLs,
vQTLs, and cQTLs in this cell type:

.. code-block:: python

   import numpy as np
   from statsmodels.stats.multitest import multipletests

   for table, prefix in ((eqtls, "de"), (vqtls, "dv"), (cqtls, "corr")):
       valid = np.isfinite(table[f"{prefix}_pval"])
       table[f"{prefix}_fdr"] = np.nan
       if valid.any():
           table.loc[valid, f"{prefix}_fdr"] = multipletests(
               table.loc[valid, f"{prefix}_pval"], method="fdr_bh"
           )[1]

These are association-level corrections over the supplied tables, not a
gene-level correction for searching multiple variants per gene. If the reported
claims span cell types or combine moment types, define the testing family
accordingly. Keep NaN results as missing and inspect their valid groups and
design; they are not evidence of no association.

Design and backend considerations
---------------------------------

Each gene-SNP or pair-SNP association is tested separately, conditional on the supplied
covariates. SNPs listed together are not automatically conditioned on one another.
``covariate_for_gene`` can similarly map each tested gene to a subset of covariate
column names. Treat dictionary keys, column names, and group order as part of the
design; memento does not infer missing genotypes or align samples by row labels.

Do not include a full set of donor indicators when testing donor-constant
variants: those indicators explain the genotype. Monomorphic SNPs are omitted
with a warning. GPU tests whose treatment is numerically explained by covariates
return NaN. Filters can remove genes before testing; use retained gene names
when constructing the dictionary.

The GPU backend shares cell resampling across genes, bootstraps each gene once,
and handles its assigned SNPs in small regression blocks. Replicate resampling
uses the same group assignments for all treatments of a gene. Different genes
can have different valid groups and designs. Wide SNP panels and replicate
resampling therefore have different performance from a single binary contrast.

This workflow uses the default ``hyper_relative`` estimator and tests both mean
and residual variability. The older ``run_eqtl`` convenience wrapper instead
uses a CPU mean-only workflow; it does not expose the GPU backend. Validation of
the GPU dictionary interface uses synthetic dosages and comparison with CPU
regression, rather than a complete real-genotype association study.
