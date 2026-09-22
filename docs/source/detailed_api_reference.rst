Implementation reference
========================

These modules implement the estimators and bootstrap machinery behind the public
API. Private helpers may change between releases; use :doc:`api_reference` for
analysis scripts. GPU implementation helpers are deliberately not imported when
building the CPU documentation.

Estimators
----------

.. automodule:: memento.estimator
   :members: _hyper_1d_relative, _hyper_cov_relative, _residual_variance

Bootstrap sampling
------------------

.. automodule:: memento.bootstrap
   :members: _bootstrap_1d, _bootstrap_2d

Regression
----------

.. automodule:: memento.hypothesis_test
   :members: _regress_1d, _regress_2d, _replicate_assignments
