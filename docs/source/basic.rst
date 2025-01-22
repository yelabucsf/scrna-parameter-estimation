Usage guide
===============
This tutorial goes over using memento to compare 2 groups of cells in a scRNA-seq dataset.

This assumes that there are no hierarchical structures in the data, such as different biological/technical replicates.

Simple binary 
---------------
Two groups of cells and no covariates:
`Notebook <https://nbviewer.org/github/yelabucsf/scrna-parameter-estimation/blob/master/tutorials/binary_testing.ipynb>`_ 


Binary testing with replicates
------------------------------
Fixed effect binary testing with multiple samples:
`Notebook <https://nbviewer.org/github/yelabucsf/scrna-parameter-estimation/blob/master/tutorials/binary_testing_replicates.ipynb>`_ 


Testing with covariates and between sample variability
------------------------------------------------------------
eQTL analysis (hierarchical bootstrap):
`Notebook <https://nbviewer.org/github/yelabucsf/scrna-parameter-estimation/blob/master/tutorials/hierarchical_bootstrap.ipynb>`_ 

Quasi-ML approach (in development)