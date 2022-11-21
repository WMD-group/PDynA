<!---  [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5553202.svg)](https://doi.org/10.5281/zenodo.5553202) --->
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

PDynA
=====

**Perovskite Dynamics Analysis** (PDynA) is a Python package for analysis of perovskite dynamics. 

<!---
Statement of need
--------
Coming soon.
--->

Code features
--------
- The input to the code is MD trajectories, currently readable formats are VASP-XDATCAR and LAMMPS .out files. The core class of PDynA is the Trajectory class.

- The structure recognition functions will automatically detect the constituent octahedral network and organic A-site molecules, and process the analysis. 

- The output is a selected set of the following properties: (pseudo-cubic) lattice parameter, octahedral distortion and tilting, time-averaged structure, A-site molecular orientation, A-site spatial displacement, radial distribution functions.

- The octahedral distortion and tilting calculation is the core feature of this package, which can quantitatively examine the dynamic behaviour of perovskite in terms of how octahedra tilt and distort, as well as the spatial correlation of these properties (Glazer notation). 


List of modules
-------

* **pdyna** library containing:
  * **core.py** Contains the core class Trajectory and related functions. 
  * **structural.py** Handles structure recognition and property calculations. 
  * **analysis.py** A collection of tools for computing and visualizing the output.
  * **io.py** The IO to input data files.


Requirements
------------

The main language is Python 3 and has been tested using Python 3.8+, with the following dependencies:
- Numpy
- Pymatgen
- MDAnalysis
- Scipy
- Sklearn
- ASE


Installation
------------

Coming soon.


License and attribution
-----------------------

Python code and original data tables are licensed under the MIT License.

Development notes
-----------------

### Bugs, features and questions
Please use the [Issue Tracker](https://github.com/aLearningMachine/PDynA/issues) to report bugs or request features in the first instance. For other queries about any aspect of the code, please contact Xia Liang by e-mail: xia.liang16@imperial.ac.uk. 


### Developer
- Xia Liang (Dept. Materials, Imperial College London)


References
----------

In preparation.
