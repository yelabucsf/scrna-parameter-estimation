"""Resource limits for tests, without importing the optional GPU dependency."""
import os
import sys


def pytest_addoption(parser):
    parser.addoption('--cpu-limit', type=int, default=2,
                     help='Maximum logical CPUs used by tests (default: 2).')


def pytest_configure(config):
    import pytest
    from threadpoolctl import threadpool_limits
    limit = config.getoption('--cpu-limit')
    if limit < 1:
        raise pytest.UsageError('--cpu-limit must be positive')
    config._memento_affinity = None
    if hasattr(os, 'sched_getaffinity') and hasattr(os, 'sched_setaffinity'):
        config._memento_affinity = os.sched_getaffinity(0)
        os.sched_setaffinity(0, sorted(config._memento_affinity)[:limit])
    keys = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
            'NUMEXPR_NUM_THREADS', 'NUMBA_NUM_THREADS')
    config._memento_environment = {key: os.environ.get(key) for key in keys}
    for key in keys:
        os.environ[key] = '1'
    config._memento_threadpools = threadpool_limits(limits=1)
    config._memento_torch_threads = None


def pytest_unconfigure(config):
    pools = getattr(config, '_memento_threadpools', None)
    if pools is not None:
        pools.restore_original_limits()
    for key, value in getattr(config, '_memento_environment', {}).items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    affinity = getattr(config, '_memento_affinity', None)
    if affinity is not None:
        os.sched_setaffinity(0, affinity)
    threads = getattr(config, '_memento_torch_threads', None)
    if threads is not None and 'torch' in sys.modules:
        sys.modules['torch'].set_num_threads(threads)
