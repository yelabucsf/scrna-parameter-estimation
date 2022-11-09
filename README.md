# memento

`memento` is a Python package for estimating the mean, variability, and gene correlation from scRNA-seq data as well as contructing a framework for hypothesis testing of differences in these parameters between groups of cells. Method-of-moments estimators are used for parameter estimation, and efficient resampling is used to construct confidence intervals and establish statistical significance.

### Installation

To install `memento`, pull the package from PyPI:

```
pip install -i https://test.pypi.org/simple/ memento
```

### Basic usage

The most basic usage for `memento` is to test for differences in mean, variability, and coexpression between two groups of cells defined by the experiment. The following tutorials demonstrate quick usage cases:

- [Analyzing the effect of exogenous IFN-B in CD14+ monocytes]()

### Advanced usage

`memento` is capable of handling experiments with multiple technical and biological replicates, such as batches/wells and different individuals respectively. The independent variable of interest can be defined at the cell level (environmental or genetic perturbation) or at a replicate level (SNPs in a population scale-study, individuals are replicates). The following tutorials demonstrate some more advanced use cases for `memento`:

- [Performing mean eQTL analysis]()