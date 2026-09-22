# memento

`memento` is a Python package for estimating the mean, variability, and gene correlation from scRNA-seq data as well as constructing a framework for hypothesis testing of differences in these parameters between groups of cells. Method-of-moments estimators are used for parameter estimation, and efficient resampling is used to construct confidence intervals and establish statistical significance.

### Installation

To install `memento`, pull the package from PyPI:

```
pip install memento-de
```

GPU installation and usage for the upcoming 0.1.3 release are documented in
[the GPU guide](docs/source/gpu.rst).

See the [user guide](https://memento.readthedocs.io/en/latest/) for installation, study designs, and complete analysis workflows.
