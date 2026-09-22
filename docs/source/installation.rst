Installation
============

CPU installation
----------------

Use a Python environment with Python 3.9 or newer, subject to the Python versions
supported by the Scanpy release you install:

.. code-block:: bash

   python -m pip install memento-de

The distribution is called ``memento-de``; import it as ``memento``. **Scanpy is
the only direct dependency** of the default installation. Its dependencies
provide the numerical libraries used by memento. CPU analysis does not require
or import PyTorch.

.. code-block:: python

   import memento

Optional GPU installation
-------------------------

Install a CUDA-enabled PyTorch build appropriate for your NVIDIA GPU and driver
using the `PyTorch installation selector <https://pytorch.org/get-started/locally/>`_.
Then install the extra from a checkout containing the GPU implementation:

.. code-block:: bash

   git clone https://github.com/yelabucsf/scrna-parameter-estimation.git
   cd scrna-parameter-estimation
   python -m pip install -e '.[gpu]'

Once 0.1.3 is published, the equivalent PyPI command is:

.. code-block:: bash

   python -m pip install 'memento-de[gpu]'

The extra declares ``torch>=2.1``; it does not choose a CUDA wheel for your
machine or install an NVIDIA driver. Installing PyTorch first follows its
platform-specific instructions and lets pip reuse a compatible installation.
Check it in the same Python environment you will use for memento:

.. code-block:: python

   import torch

   print(torch.__version__)
   print(torch.cuda.is_available())  # must be True
   print(torch.cuda.get_device_name(0))

Choose ``backend="gpu"`` explicitly when testing moments. The default remains
``backend="cpu"`` even when PyTorch is installed. See :doc:`gpu` for supported
options and memory management.

Development and documentation
-----------------------------

For a CPU-only source installation, use ``python -m pip install -e .``.
To build this documentation from the repository root:

.. code-block:: bash

   python -m pip install -e .
   python -m pip install -r docs/requirements.txt
   python -m sphinx -W --keep-going -b html docs/source docs/build/html

The documentation build does not need PyTorch or GPU hardware. Tests and CPU
resource limits are described in ``tests/README.md``.
