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

GPU tests construct their own synthetic counts and designs. They do not require
benchmark JSON files or external datasets. Coverage includes 1D and correlation
moments, CPU/GPU regression agreement, eQTL dictionaries, memory-limited
sampling, reproducibility, optional dependencies, and failure cleanup.
