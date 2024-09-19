.. _installation:

Installation
==============

Download and Install
--------------------

``PDynA`` can be installed with the following commands:

.. code-block:: bash

   git clone https://github.com/WMD-group/PDynA.git  # Clone the repository (or download manually)
   cd pdyna  # cd to PDynA directory with the setup.py file
   pip install .  # Install the package with pip

Note that if you already have all the dependencies installed in your environment (namely ``numpy``, ``scipy``,
``pymatgen``, ``matplotlib``, and ``ASE``), you can also install ``PDynA`` without updating these dependencies
as it only requires their fundamental functionality. For example, instead do:

.. code-block:: bash

   pip install --no-deps .  # Install the package with pip, and without changing its dependency packages

Install from PyPI
--------------------

``PDynA`` can be installed directly from PyPI:

.. code-block:: bash

   pip install pdyna  
