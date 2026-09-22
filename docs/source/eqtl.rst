Gene-specific treatments and eQTLs
==================================

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

Each gene-SNP association is tested separately, conditional on the supplied
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
