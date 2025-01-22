Welcome to memento’s documentation!
====================================================

``memento`` is a python package for performing differential mean, variability, and correlation in single-cell RNA sequencing data. Our paper can be found at "Method of moments framework for differential expression analysis of single-cell RNA sequencing data."

Some current limitations include:
- continous covariates, such as cell state variables
- non-linear representations of covariates 

**Installation**

Make sure you have a version of ``scanpy > 1.3``. Almost any version should work with ``memento``. Visit the ``scanpy`` `website <https://scanpy.readthedocs.io/en/stable/installation.html>`_  to install the latest version of scanpy. 

To install memento, run:

.. code-block:: bash

    pip install memento
.. toctree::
   :maxdepth: 2
   :caption: User guide & tutorial:

   basic

.. toctree::
   :maxdepth: 2
   :caption: About memento:

   estimators
   inference

.. toctree::
   :maxdepth: 1
   :caption: API Reference:

   api_reference

.. .. toctree::
..    :maxdepth: 1
..    :caption: Detailed API Reference:
   
..    detailed_api_reference

Indices and tables
==================
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`