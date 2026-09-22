GPU acceleration
================

Install the optional dependency as described in :doc:`installation`. Prepare
counts, groups, and observed moments using the usual functions, then select the
backend at testing time:

.. code-block:: python

   memento.ht_1d_moments(
       adata, treatment=treatment, covariate=covariate,
       backend="gpu", num_boot=10000, random_state=5,
       approx="norm", resample_rep=False,
   )
   results = memento.get_1d_ht_result(adata)

Both ``resample_rep=False`` and ``resample_rep=True`` are supported for 1D and 2D
tests, including gene-specific and pair-specific treatment/covariate dictionaries.
Choose replicate resampling based on the design; see :doc:`inference`.

What runs on the GPU
--------------------

The backend samples individual cells jointly across genes, avoiding per-gene
expression-state compression. Bootstrap moments and regression run on CUDA.
Replicate resampling also uses CUDA-generated group and iteration indices with
bounded batches of regression work. Input preparation, observed moments, and
covariate projection setup still run on CPU.

Moments use float32 with TF32 disabled; transforms and regressions use float64.
CPU and GPU draws differ, even for the same seed. Their numerical and Monte Carlo
differences should be distinguished from a change in the statistical procedure.
Tests compare fixed-weight moments and regression with identical bootstrap
inputs and group assignments, in addition to public API behavior.

Memory and device settings
--------------------------

The defaults adapt to available device memory:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Argument
     - Behavior
   * - ``gpu_memory_budget=None``
     - Target half the currently free GPU memory, capped at 8 GiB.
   * - ``gpu_memory_budget=<integer>``
     - Target this many MiB, capped at 75% of currently free GPU memory.
   * - ``gpu_batch_size=256``
     - Cap simultaneous genes (1D) or pairs (2D); the planner may lower it.
   * - ``gpu_device="cuda"``
     - Select the default CUDA device; use e.g. ``"cuda:1"`` for another GPU.

These are working-memory targets, excluding CUDA context and allocator cache,
not hard limits on process memory. Cell weights are cached when they fit and
otherwise regenerated in chunks. Replicate regression bounds treatment and
draw temporaries and includes its additional workspace in memory planning.
For a lower-memory device, start with the defaults; reduce the budget or batch
cap if sharing a busy GPU. Another process can consume memory after planning.
More memory can permit caching or larger batches, but does not guarantee a
speedup when computation or kernel overhead dominates.

Inspect the plan used for a completed call:

.. code-block:: python

   print(adata.uns["memento"]["1d_ht"]["gpu"])
   # For correlation tests, use ["2d_ht"]["gpu"].

Support and portability
-----------------------

GPU testing currently requires integer-valued, nonnegative counts, the
``hyper_relative`` estimator, and ``approx="norm"``. Unsupported options raise
errors rather than silently switching to CPU. CPU remains the default and does
not import PyTorch.

The implementation uses standard PyTorch operations rather than kernels tied to
one GPU model. CUDA-compatible NVIDIA GPUs are the current target. Hardware
validation has used an RTX 3060 with 12 GiB memory; memory planning and streaming
are also tested with small budgets. Other GPU models have not been hardware
validated. Apple MPS is unsupported and AMD/ROCm is unvalidated. FP64 throughput
and available memory vary between devices, so performance will vary too.

For complete examples, see :doc:`basic`, :doc:`correlations`, and :doc:`eqtl`.
