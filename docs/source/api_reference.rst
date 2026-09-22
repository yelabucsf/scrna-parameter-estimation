Public API
==========

Functions are available from ``import memento``. Start with :doc:`basic` for the
required order of operations and :doc:`designs` for constructing numeric designs.

Preparation and moments
-----------------------

.. autofunction:: memento.main.setup_memento
.. autofunction:: memento.main.create_groups
.. autofunction:: memento.main.get_groups
.. autofunction:: memento.main.compute_1d_moments
.. autofunction:: memento.main.compute_2d_moments
.. autofunction:: memento.main.get_1d_moments
.. autofunction:: memento.main.get_2d_moments
.. autofunction:: memento.main.get_corr_matrix

Hypothesis testing
------------------

.. autofunction:: memento.main.ht_1d_moments
.. autofunction:: memento.main.ht_2d_moments
.. autofunction:: memento.main.get_1d_ht_result
.. autofunction:: memento.main.get_2d_ht_result

Other CPU analysis functions
----------------------------

``ht_mean`` provides the separate quasi-ML mean-testing path; it does not use
the GPU moment-testing backend described in this guide.

.. autofunction:: memento.main.ht_mean
.. autofunction:: memento.main.get_mean_ht_result

CPU convenience wrappers
------------------------

.. autofunction:: memento.wrappers.binary_test_1d
.. autofunction:: memento.wrappers.binary_test_2d
.. autofunction:: memento.wrappers.run_eqtl
