# Final GPU test validation

Validated 2026-09-22 in the existing `torch` conda environment, Python 3.14,
PyTorch 2.9.0+cu128, RTX 3060 12 GB. Execution was constrained to at most ten
logical CPUs. Pytest 9.1.1 was installed under `/tmp/memento-test-deps`; the
conda environment was not modified.

| Configuration | Passed | Skipped | Failed |
|---|---:|---:|---:|
| Full suite with CUDA available | 76 | 0 | 0 |
| Full suite with PyTorch imports blocked | 51 | 25 | 0 |

All 48 pre-existing tests pass. The 28 cases in `tests/test_gpu.py` cover
sampling, fixed-weight moments and correlations, CPU/GPU regression agreement,
wide eQTL dictionaries, treatment ordering and filtering, identifiability,
reproducibility, optional dependencies, invalid inputs and unsupported options,
memory planning and streaming, shared draws across gene batches, diagonal and
invalid-population cases, and restoration of CUDA state after exceptions.

The no-PyTorch run uses an import hook raising `ModuleNotFoundError` for torch;
it is not a newly provisioned environment. It verifies CPU execution without
loading torch and clean skips for GPU-dependent tests. A subprocess also checks
that explicitly importing the GPU module without torch gives installation
instructions.

Generated distribution metadata was checked directly: the sole unconditional
requirement is `scanpy`; `torch>=2.1` is conditioned on `extra == "gpu"`.
`git diff --check` passed.

There were six warnings in the CUDA run: two poorly conditioned polynomial
fits in existing tests, two divide-by-zero warnings in an existing sparse-group
error test, the expected monomorphic-treatment warning, and PyTorch's TF32 API
deprecation warning. No test failed or was marked xfail. The no-PyTorch run has
only the four pre-existing numerical warnings.

Resource limits now live in `tests/conftest.py`; the suite also passed after
removing the experimental `runtime.py`. Use `--cpu-limit=10` to override the
default two-CPU test limit.

The ordinary suite command is `python -m pytest tests -q` with pytest installed
in the selected environment. CUDA is needed to exercise all 76 cases; GPU cases
skip when the optional dependency or hardware is unavailable.

- [CUDA JUnit results](results_final_tests/cuda.xml)
- [No-PyTorch JUnit results](results_final_tests/no_torch.xml)
- [PBMC full-panel GPU validation](REAL_DATA_GPU_RESULTS.md)
- [Correlation validation](CORRELATION_RESULTS.md)
- [Full-panel fibroblast CPU/GPU comparison](FIBROBLAST_RESULTS.md)

The real-data experiments above were completed previously and were not repeated
for this final test run; no backend implementation changes were required.
