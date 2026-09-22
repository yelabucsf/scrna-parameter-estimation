memento: differential expression beyond the mean
================================================

memento estimates and tests differences in **mean expression**, **variability**,
and **gene correlation** from single-cell RNA sequencing counts. Its method of
moments estimators account for RNA capture noise; bootstrap samples quantify
uncertainty in the estimated effects.

Start with :doc:`installation` and the :doc:`basic` workflow. For multi-sample
experiments, read :doc:`designs` before choosing a resampling scheme. The
:doc:`gpu` backend accelerates both bootstrap sampling and regression while
keeping the same public analysis functions.

.. note::

   These pages describe the 0.1.3 code, including optional GPU acceleration and
   GPU replicate resampling. Until this version is published on PyPI, install
   from a checkout containing these changes.

.. toctree::
   :maxdepth: 2
   :caption: Get started

   installation
   basic
   designs
   correlations
   eqtl
   gpu

.. toctree::
   :maxdepth: 2
   :caption: Methods and troubleshooting

   estimators
   inference
   troubleshooting
   release_notes

.. toctree::
   :maxdepth: 1
   :caption: Reference

   api_reference
   detailed_api_reference

Source code and issue reports are on
`GitHub <https://github.com/yelabucsf/scrna-parameter-estimation>`_.
The repository's ``publication/README.md`` describes how to reproduce the paper,
*Method of moments framework for differential expression analysis of
single-cell RNA sequencing data*.
