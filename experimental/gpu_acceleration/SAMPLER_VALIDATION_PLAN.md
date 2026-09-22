# Staged custom sampler experiment

A separate [shared cell-weight approach](CELL_MATMUL_RESULTS.md) has now been
tested using existing integer RNG operations and matrix multiplication. It is
faster on the current many-gene workload. This plan still applies to a future
custom conditional-binomial kernel; that kernel is not integrated.

Standalone probability-cache wrappers have been tested, with no useful measured
speedup; see [SAMPLER_CACHE_RESULTS.md](SAMPLER_CACHE_RESULTS.md). Keep the validated PyTorch sampler
as the reference while testing a standalone candidate. The goal is to reduce
repeated sampling work, not change the multinomial bootstrap distribution.

The initial [cache feasibility probe](SAMPLER_CACHE_PROBE.md) is complete:
probability/count inputs show substantial reuse, with shared tables fitting in
memory. Its recommended probability-only benchmark is now complete. Larger
count-dependent tables remain untested.

1. **Specify the algorithm before implementing it.** Document the binomial
   method, numerical precision, RNG stream assignment and supported count and
   probability ranges. Start with a narrow implementation and explicit fallback.
   Do not substitute Poisson draws or revive the rejected CDF-inversion approach.
2. **Separate arithmetic from random sampling.** Feed identical predetermined
   counts through both moment accumulators. Check remaining-count updates,
   per-row capture rates, padding and final normalization independently of RNG.
3. **Validate binomial draws in isolation.** Cover zero/one probabilities,
   zero/one trials, tiny probabilities, values near one, symmetric probabilities
   and counts spanning the real workload. Compare small cases against exact
   probability masses and examine tails as well as means and variances.
4. **Validate the joint multinomial distribution.** Require nonnegative integer
   counts summing to each row's cell count. Check marginal means, variances and
   negative cross-state covariances, including heterogeneous rows and padding.
   Test fresh draws and independence across rows, bootstrap draws and repeated
   invocations. Check replay and buffer aliasing if caching is introduced.
5. **Use prespecified statistical tolerances.** Choose sample sizes, seeds and
   acceptance thresholds from Monte Carlo uncertainty, accounting for multiple
   comparisons. Run the existing sampler through the same checks as a control.
   Do not loosen thresholds after observing a candidate failure.
6. **Integrate only after standalone checks pass.** Compare real-data mean and
   variability estimates, standard errors and p-values against repeated reference
   runs. Separately compare CPU and GPU regressions on identical draws. Exact
   draw equality is not expected when the RNG algorithm changes.
7. **Measure the complete cost.** Include setup/table construction, transfers,
   cold and warm calls, and peak memory. Retain the candidate only if repeated
   full-workload trials demonstrate useful speed gains with passing validation.

An initial checkpoint should present the proposed algorithm and standalone
correctness evidence before replacing the sampler in the experimental pipeline.
Statistical tests support implementation confidence; they do not prove a sampler
is exact. Algorithm review and numerical analysis remain necessary.
