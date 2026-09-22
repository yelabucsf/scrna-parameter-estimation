GPU acceleration
================

.. note::

   This page documents the GPU backend planned for the 0.1.3 release.
   Until that release is published, install from a checkout containing these
   changes rather than expecting the GPU extra in an older PyPI release.

CPU installation
----------------

.. code-block:: bash

   pip install memento-de

Scanpy is the only direct dependency of the CPU installation; its dependencies
provide the numerical libraries used by memento. PyTorch is not required or
imported for CPU use. Install the optional ``gpu`` extra only for GPU execution.

GPU installation
----------------

From a checkout containing the GPU implementation, install ``pip install -e '.[gpu]'`` into an environment with a
CUDA-enabled PyTorch build. After the usual ``setup_memento``, ``create_groups``,
and ``compute_1d_moments`` steps, use:

.. code-block:: python

   memento.ht_1d_moments(
       adata, treatment=treatment, covariate=covariate,
       backend="gpu", num_boot=10000, random_state=5,
       approx="norm", resample_rep=False,
   )
   results = memento.get_1d_ht_result(adata)


Sampling and reproducibility
----------------------------

Treatment and covariate rows must follow ``adata.uns['memento']['groups']``.
The GPU backend samples cells jointly across genes and computes bootstrap
moments and regressions on CUDA, without expression-state compression. It
retains the existing size-factor approximation and per-gene bootstrap
distribution; bootstrap draws are shared across genes and differ from CPU
draws. A fixed seed and execution configuration are reproducible, but changing
batching or memory settings can change the draws.

Memory and device settings
--------------------------

``gpu_memory_budget=None`` (default) automatically targets half of currently free
GPU memory, capped at 8 GiB. An explicit integer sets a target in MiB, capped at
75% of free memory. Targets exclude CUDA context and allocator cache; this is
not a hard allocation limit or protection against another process consuming
memory after planning.
``gpu_batch_size`` caps simultaneous genes; the memory planner may lower it.
Cell weights are cached when they fit, otherwise regenerated in chunks.
Moments use float32 with TF32 disabled; transforms and regressions use float64.
Design projections and data preparation still run on CPU. ``gpu_device`` selects
the CUDA device (default ``"cuda"``). The implementation uses standard PyTorch
operations rather than kernels tied to the RTX 3060. Only that GPU has been
hardware-tested so far; low-memory behavior and planning for different free
memory sizes are tested separately. CUDA-compatible NVIDIA GPUs are the
current target; Apple MPS is unsupported and AMD/ROCm is unvalidated.

Supported options
-----------------

Currently supported: ``hyper_relative``, ``approx="norm"``, and
``resample_rep=False``, including gene-specific treatments/covariates.
Other estimators and replicate resampling are not supported by this backend.
CPU remains the default and does not import PyTorch.

Gene-specific treatments (eQTLs)
--------------------------------

For eQTL-style testing, supply genotype dosage columns in ``treatment`` and select
columns per gene using a dictionary:

.. code-block:: python

   memento.ht_1d_moments(
       adata, treatment=genotypes, covariate=covariates,
       treatment_for_gene={"GENE_A": ["rs1", "rs2"], "GENE_B": ["rs3"]},
       backend="gpu", num_boot=10000, random_state=5,
   )


Rows must match group order; dictionary keys must be retained genes. These are
separate covariate-adjusted tests, not a joint model conditioning each SNP on
all the other SNPs. Each gene is bootstrapped once and its assigned treatments
are processed in small blocks to bound regression memory. Different treatment
sets can reduce cross-gene regression batching, so timing depends on the design.
``covariate_for_gene`` optionally selects covariates per gene. Monomorphic SNPs
are omitted; SNPs numerically explained by the covariates return NaN on GPU.
Do not include full donor indicators when testing donor-constant genotype main
effects: those indicators explain the genotype. No real genotype dataset was
used in the current validation; dictionary behavior and wide designs are tested
with synthetic dosages against CPU regression.

Differential correlation
------------------------

For differential correlation, supply an explicit list of gene pairs after
computing 1D moments, then select the same GPU backend:

.. code-block:: python

   memento.compute_2d_moments(adata, gene_pairs)
   memento.ht_2d_moments(
       adata, treatment=treatment, covariate=covariate,
       backend="gpu", num_boot=10000, random_state=5,
       approx="norm", resample_rep=False,
   )
   correlations = memento.get_2d_ht_result(adata)


Here ``gpu_batch_size`` counts pairs. Shared cell weights jointly generate both
variances and covariance for each pair. The backend preserves the CPU handling
of invalid correlations and regression on raw correlations; reported effects
are observed correlation differences adjusted for the covariates. Self-pairs
return NaN. Pair-specific treatment/covariate selections are supported by the
GPU backend. Observed moments are still prepared on CPU.
