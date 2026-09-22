# Running tests

Install the package and pytest in your chosen environment, then run:

```bash
python -m pytest tests -q
python -m pytest tests -q --cpu-limit=10
```

`conftest.py` defaults to at most two logical CPUs on platforms with process
affinity support. Numerical library threads are restricted to one to avoid
nested parallelism. GPU tests also restrict PyTorch CPU threads, without
importing PyTorch for CPU-only tests. Original settings are restored at exit.
These controls apply only during pytest; importing memento has no resource
configuration side effects.

Install a compatible CUDA-enabled PyTorch build and `pip install -e '.[gpu]'`
to exercise the GPU cases. They skip when PyTorch or CUDA is unavailable.

The standalone experiment runners do not configure global CPU affinity. Apply
resource limits explicitly when reproducing benchmarks, for example on Linux:

```bash
taskset -c 0-9 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python experimental/gpu_acceleration/run_fibroblasts.py
```

Choose CPU IDs allowed by your environment. The fibroblast runner uses ten
joblib threads and limits BLAS threads during its CPU/GPU test calls.
