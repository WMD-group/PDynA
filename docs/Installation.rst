.. _installation:

Installation
==============

Download and Install
--------------------

``PDynA`` can be installed with the following commands:

.. code-block:: bash

   git clone https://github.com/WMD-group/PDynA.git
   cd pdyna
   pip install .

Note that if you already have all the dependencies installed in your environment (namely ``numpy``, ``scipy``,
``pymatgen``, ``matplotlib``, and ``ASE``), you can also install ``PDynA`` without updating these dependencies
as it only requires the very basic functionality of them. For example, do:

.. code-block:: bash

   pip install --no-deps .

Install from PyPI (not yet available)
-------------

``PDynA`` can be installed directly from PyPI:

.. code-block:: bash

   pip install pdyna  
