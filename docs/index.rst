
PDynA
=======================================

.. image:: https://github.com/WMD-group/PDynA/blob/main/graphic.png
   :target: https://github.com/WMD-group/PDynA

``PDynA`` is an open-source Python package for computing structural properties of perovskites from molecular dynamics output.

Key Features
============
All features and functionality are fully-customisable:

- **Processing of MD Trajectory**: Read from most of the common MD software outputs.
- **Structural Analysis**: Compute structural properties of the perovskite structure including (local) lattice parameters, time-averaged structure, molecular motion, site displacements, etc.
- **Octahedral deformation**: Compute octahedral tilting and distortion, giving direct information on phase transformation and dynamic ordering.
- **Spatial and Time Correlations**: Calculate the correlation of various properties in space and time, with customizable functions. 
- **Vectorization and Parallelization**: Most of the functions are vectorized and/or can be parallelized with multi-threading, under an almost perfect scaling. 
- **Plotting**: Generate plots of computed outputs, providing visual intuition on the key properties, and can also be easily tuned for your needs. 
- ``Python`` **Interface**: Customisable and modular ``Python`` API. 

Citation
========

If you use ``PDynA`` in your research, please cite:

- Xia Liang et al. `Structural Dynamics Descriptors for Metal Halide Perovskites <https://doi.org/10.1021/acs.jpcc.3c03377>`__. *The Journal of Physical Chemistry C* 127 (38), 19141-19151, **2023**

.. toctree::
   :hidden:
   :caption: Usage
   :maxdepth: 4

   Installation
   Python API <pdyna>
   Tutorials
   Tips

.. toctree::
   :hidden:
   :caption: Information
   :maxdepth: 1

   Contributing
   Code_Compatibility
   PDynA on GitHub <https://github.com/WMD-group/PDynA>
