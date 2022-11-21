# PDynA
PDynA is a Python package for analysis of perovskite dynamic. 
The input to the code is MD trajectory, currently readable formats are VASP-XDATCAR and LAMMPS .out files. 
The output is a selected set of the following properties: (pseudo-cubic) lattice parameter, octahedral distortion and tilting, time-averaged structure, A-site molecular orientation, A-site spatial displacement, radial distribution functions. 
Dependencies: Numpy, Pymatgen, MDAnalysis, Scipy, Sklearn, ASE. 
