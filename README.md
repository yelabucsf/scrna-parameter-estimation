# memento

`memento` is a Python package for estimating the mean, variability, and gene correlation from scRNA-seq data as well as contructing a framework for hypothesis testing of differences in these parameters between groups of cells. Method-of-moments estimators are used for parameter estimation, and efficient resampling is used to construct confidence intervals and establish statistical significance.

### Installation

To install `memento`, pull the package from PyPI:

```
pip install memento-de
```

Memento only has one dependency, which is `scanpy`. Any version of python > 3.9 should work.

For more information, please refer to the documentation [here](https://memento.readthedocs.io/en/master/index.html)!
