"""Core objects of PDyna. dev1"""

from __future__ import annotations

from dataclasses import dataclass, field
from pymatgen.io.ase import AseAtomsAdaptor as aaa
import matplotlib.pyplot as plt
from pdyna.io import print_time
from pdyna.structural import distance_matrix_handler
import numpy as np
import time
import os


@dataclass
class Trajectory:
    
    """
    Main class representing the MD trajectory to analyze.
    Initialize the class with reading the raw data.

    Parameters
    ----------
    data_format : data format based on the MD software
        Currently compatible formats are 'vasp', 'xyz', 'ase-traj', 'pdb' and 'lammps'.
    data_path : tuple of input files
        The input file path.
        vasp: (poscar_path, xdatcar_path, incar_path)
        lammps: (dump.out_path, MD setting tuple)
        xyz: (xyz_path, MD setting tuple)
        ase-trajectory: (ase_traj_path, MD setting tuple)
        pdb: (pdb_path, MD setting tuple)
        
            format for MD setting tuple: (Ti,Tf,tstep) , this is the same for all formats except VASP which uses INCAR file
    """
    
    data_format: str = field(repr=False)
    data_path: tuple = field(repr=False)
    
    _Xsite_species = ['Cl','Br','I','Se'] # update if your structrue contains other elements on the X-sites
    _Bsite_species = ['Pb','Sn','W'] # update if your structrue contains other elements on the B-sites
    _known_elem = ("I", "Br", "Cl", "Pb", "C", "H", "N", "Cs", "Se", "W", "Sn") # update if your structure has other constituent elements, this is just to make sure all the elements should appear in the structure. 
    
    # characteristic value of bond length of your material for structure construction 
    # the first interval should be large enough to cover all the first and second NN of B-X (B-B) pairs, 
    # in the second list, the two elements are 0) approximate first NN distance of B-X (B-B) pairs, and 1) 0) approximate second NN distance of B-X (B-B) pairs
    # These values can be obtained by inspecting the initial configuration or, e.g. in the pair distrition function of the structure
    _fpg_val_BB = [[3,10], [6.3,9.1]] # empirical values for lead halide perovskites
    _fpg_val_BX = [[0.1,8], [3,6.8]] # empirical values for lead halide perovskites
    #_fpg_val_BB = [[2,6], [3.2,5.7]] # WSe2
    #_fpg_val_BX = [[1,4.6], [2.5,4.1]] # WSe2
    
    def __post_init__(self):
        
        et0 = time.time()

        if self.data_format == 'vasp':
            
            import pymatgen.io.vasp.inputs as vi
            from pdyna.io import chemical_from_formula, read_xdatcar
            
            if len(self.data_path) != 3:
                raise TypeError("The input format for vasp must be (poscar_path, xdatcar_path, incar_path). ")
            poscar_path, xdatcar_path, incar_path = self.data_path
            
            print("------------------------------------------------------------")
            print("Loading Trajectory files...")
            
            # read POSCAR and XDATCAR files
            st0 = vi.Poscar.from_file(poscar_path,check_for_POTCAR=False).structure # initial configuration
            
            for elem in st0.symbol_set:
                if not elem in self._known_elem:
                    raise ValueError(f"An unexpected element {elem} is found. Please update the list known_elem above. ")
            self.st0 = st0
            self.natom = len(st0)
            self.species_set = st0.symbol_set
            self.formula = chemical_from_formula(st0)

            atomic_symbols, self.lattice, self.latmat, self.Allpos = read_xdatcar(xdatcar_path,self.natom)
            self.nframe = self.lattice.shape[0]
            
            if atomic_symbols != [site.species_string for site in self.st0.sites]:
                raise TypeError("The atomic species in the POSCAR does not match with those in the trajectory. ")
                
            # read the coordinates and save   
            Xindex = []
            Bindex = []
            Cindex = []
            Nindex = []
            Hindex = []
            for i,site in enumerate(atomic_symbols):
                 if site in self._Xsite_species:
                     Xindex.append(i)
                 if site in self._Bsite_species:
                     Bindex.append(i)  
                 if site == 'C':
                     Cindex.append(i)  
                 if site == 'N':
                     Nindex.append(i)  
                 if site == 'H':
                     Hindex.append(i)  
            
            self.Bindex = Bindex
            self.Xindex = Xindex
            self.Cindex = Cindex
            self.Hindex = Hindex
            self.Nindex = Nindex
            
            # read INCAR to obatin MD settings
            if type(incar_path) == str:
                with open(incar_path,"r") as fp:
                    lines = fp.readlines()
                    nblock = 1
                    Tf = None
                    for line in lines:
                        if line.startswith('NBLOCK'):
                            nblock = int(line.split()[2])
                        if line.startswith('TEBEG'):
                            Ti = float(line.split()[2])
                            if Tf == None:
                                Tf = Ti
                        if line.startswith('TEEND'):
                            Tf = int(line.split()[2])
                        if line.startswith('POTIM'):
                            tstep = float(line.split()[2])
                        if line.startswith('NSW'):
                            nsw = int(line.split()[2])
            elif type(incar_path) == tuple:
                Ti, Tf, tstep, nsw, nblock = incar_path
                
            else:
                raise TypeError("The MD settings must be in the form of either an INCAR file or a tuple of (Ti, Tf, tstep, nsw, nblock).")
            
            self.MDsetting = {}
            self.MDsetting["nsw"] = nsw
            self.MDsetting["nblock"] = nblock
            self.MDsetting["Ti"] = Ti
            self.MDsetting["Tf"] = Tf
            self.MDsetting["tstep"] = tstep
            self.MDTimestep = tstep/1000*nblock  # the timestep between recorded frames
            self.Tgrad = (Tf-Ti)/(nsw*tstep/1000)   # temeperature gradient
        
        
        elif self.data_format == 'lammps':
            
            from pdyna.io import read_lammps_dump, chemical_from_formula
            
            if len(self.data_path) == 2:
                with_init = False
                dump_path, lammps_setting = self.data_path  
            elif len(self.data_path) == 3:
                with_init = True
                dump_path, lammps_setting, initial_path = self.data_path  
            else:
                raise TypeError("The input format for lammps must be either (dump.out_path, MD setting tuple) or (dump.out_path, MD setting tuple, lammps.data_path). ")
              
            
            print("------------------------------------------------------------")
            print("Loading Trajectory files...")
            
            if len(lammps_setting) == 3:
                atomic_symbols, lattice, latmat, Allpos, st0, max_step, stepsize = read_lammps_dump(dump_path)
            elif len(lammps_setting) == 4:
                atomic_symbols, lattice, latmat, Allpos, st0, max_step, stepsize = read_lammps_dump(dump_path,lammps_setting[3])
            
            if with_init:
                at0 = aaa.get_atoms(st0)
                atsym = at0.numbers
                from ase.io.lammpsdata import read_lammps_data
                atn = read_lammps_data(initial_path,style='atomic')
                assert len(at0) == len(atn)
                atn.numbers = atsym
                st0 = aaa.get_structure(atn)
            
            for elem in st0.symbol_set:
                if not elem in self._known_elem:
                    raise ValueError(f"An unexpected element {elem} is found. Please check the list known_elem. ")
             
            self.st0 = st0
            self.natom = len(st0)
            self.species_set = st0.symbol_set
            self.formula = chemical_from_formula(st0)
            
            Xindex = []
            Bindex = []
            Cindex = []
            Nindex = []
            Hindex = []
            for i,site in enumerate(atomic_symbols):
                 if site in self._Xsite_species:
                     Xindex.append(i)
                 if site in self._Bsite_species:
                     Bindex.append(i)  
                 if site == 'C':
                     Cindex.append(i)  
                 if site == 'N':
                     Nindex.append(i)  
                 if site == 'H':
                     Hindex.append(i)  
            
            self.Bindex = Bindex
            self.Xindex = Xindex
            self.Cindex = Cindex
            self.Hindex = Hindex
            self.Nindex = Nindex
            
            self.Allpos = Allpos
            
            self.lattice = lattice
            self.latmat = latmat
            
            self.nframe = lattice.shape[0]
            
            
            self.MDsetting = {}
            self.MDsetting["nsw"] = max_step
            self.MDsetting["nblock"] = stepsize
            self.MDsetting["Ti"] = lammps_setting[0]
            self.MDsetting["Tf"] = lammps_setting[1]
            self.MDsetting["tstep"] = lammps_setting[2]
            self.MDTimestep = lammps_setting[2]/1000*stepsize  # the timestep between recorded frames
            self.Tgrad = (lammps_setting[1]-lammps_setting[0])/(stepsize*lammps_setting[2])   # temeperature gradient
        
        
        elif self.data_format == 'xyz':
            
            from pdyna.io import read_xyz, chemical_from_formula
            
            if len(self.data_path) != 2:
                raise TypeError("The input format for lammps must be (xyz_path, MD setting tuple). ")
            xyz_path, md_setting = self.data_path    
            
            print("------------------------------------------------------------")
            print("Loading Trajectory files...")
            
            atomic_symbols, lattice, latmat, Allpos, st0, nstep = read_xyz(xyz_path)
            
            for elem in st0.symbol_set:
                if not elem in self._known_elem:
                    raise ValueError(f"An unexpected element {elem} is found. Please check the list known_elem. ")
             
            self.st0 = st0
            self.natom = len(st0)
            self.species_set = st0.symbol_set
            self.formula = chemical_from_formula(st0)
            
            Xindex = []
            Bindex = []
            Cindex = []
            Nindex = []
            Hindex = []
            for i,site in enumerate(atomic_symbols):
                 if site in self._Xsite_species:
                     Xindex.append(i)
                 if site in self._Bsite_species:
                     Bindex.append(i)  
                 if site == 'C':
                     Cindex.append(i)  
                 if site == 'N':
                     Nindex.append(i)  
                 if site == 'H':
                     Hindex.append(i)  
            
            self.Bindex = Bindex
            self.Xindex = Xindex
            self.Cindex = Cindex
            self.Hindex = Hindex
            self.Nindex = Nindex
            
            self.Allpos = Allpos
            
            self.lattice = lattice
            self.latmat = latmat
            
            self.nframe = lattice.shape[0]
            
            
            self.MDsetting = {}
            self.MDsetting["nblock"] = md_setting[3]
            self.MDsetting["nsw"] = nstep*md_setting[3]
            self.MDsetting["Ti"] = md_setting[0]
            self.MDsetting["Tf"] = md_setting[1]
            self.MDsetting["tstep"] = md_setting[2]
            self.MDTimestep = md_setting[2]/1000*md_setting[3]  # the timestep between recorded frames
            self.Tgrad = (md_setting[1]-md_setting[0])/(md_setting[3]*md_setting[2]/1000)   # temeperature gradient
            
        
        elif self.data_format == 'pdb':
            
            from pdyna.io import read_pdb, chemical_from_formula
            
            if len(self.data_path) != 2:
                raise TypeError("The input format for lammps must be (pdb_path, MD setting tuple). ")
            pdb_path, md_setting = self.data_path    
            
            print("------------------------------------------------------------")
            print("Loading Trajectory files...")
            
            atomic_symbols, lattice, latmat, Allpos, st0, nstep = read_pdb(pdb_path)
            
            for elem in st0.symbol_set:
                if not elem in self._known_elem:
                    raise ValueError(f"An unexpected element {elem} is found. Please check the list known_elem. ")
             
            self.st0 = st0
            self.natom = len(st0)
            self.species_set = st0.symbol_set
            self.formula = chemical_from_formula(st0)
            
            Xindex = []
            Bindex = []
            Cindex = []
            Nindex = []
            Hindex = []
            for i,site in enumerate(atomic_symbols):
                 if site in self._Xsite_species:
                     Xindex.append(i)
                 if site in self._Bsite_species:
                     Bindex.append(i)  
                 if site == 'C':
                     Cindex.append(i)  
                 if site == 'N':
                     Nindex.append(i)  
                 if site == 'H':
                     Hindex.append(i)  
            
            self.Bindex = Bindex
            self.Xindex = Xindex
            self.Cindex = Cindex
            self.Hindex = Hindex
            self.Nindex = Nindex
            
            self.Allpos = Allpos
            
            self.lattice = lattice
            self.latmat = latmat
            
            self.nframe = lattice.shape[0]
            
            
            self.MDsetting = {}
            self.MDsetting["nblock"] = md_setting[3]
            self.MDsetting["nsw"] = nstep*md_setting[3]
            self.MDsetting["Ti"] = md_setting[0]
            self.MDsetting["Tf"] = md_setting[1]
            self.MDsetting["tstep"] = md_setting[2]
            self.MDTimestep = md_setting[2]/1000*md_setting[3]  # the timestep between recorded frames
            self.Tgrad = (md_setting[1]-md_setting[0])/(md_setting[3]*md_setting[2]/1000)   # temeperature gradient


        elif self.data_format == 'ase-traj':
            
            from pdyna.io import read_ase_traj, chemical_from_formula
            
            if len(self.data_path) != 2:
                raise TypeError("The input format for lammps must be (ase_traj_path, MD setting tuple). ")
            ase_path, md_setting = self.data_path    
            
            print("------------------------------------------------------------")
            print("Loading Trajectory files...")
            
            atomic_symbols, lattice, latmat, Allpos, st0, nstep = read_ase_traj(ase_path)
            
            for elem in st0.symbol_set:
                if not elem in self._known_elem:
                    raise ValueError(f"An unexpected element {elem} is found. Please check the list known_elem. ")
             
            self.st0 = st0
            self.natom = len(st0)
            self.species_set = st0.symbol_set
            self.formula = chemical_from_formula(st0)
            
            Xindex = []
            Bindex = []
            Cindex = []
            Nindex = []
            Hindex = []
            for i,site in enumerate(atomic_symbols):
                 if site in self._Xsite_species:
                     Xindex.append(i)
                 if site in self._Bsite_species:
                     Bindex.append(i)  
                 if site == 'C':
                     Cindex.append(i)  
                 if site == 'N':
                     Nindex.append(i)  
                 if site == 'H':
                     Hindex.append(i)  
            
            self.Bindex = Bindex
            self.Xindex = Xindex
            self.Cindex = Cindex
            self.Hindex = Hindex
            self.Nindex = Nindex
            
            self.Allpos = Allpos
            
            self.lattice = lattice
            self.latmat = latmat
            
            self.nframe = lattice.shape[0]
            
            
            self.MDsetting = {}
            self.MDsetting["nblock"] = md_setting[3]
            self.MDsetting["nsw"] = nstep*md_setting[3]
            self.MDsetting["Ti"] = md_setting[0]
            self.MDsetting["Tf"] = md_setting[1]
            self.MDsetting["tstep"] = md_setting[2]
            self.MDTimestep = md_setting[2]/1000*md_setting[3]  # the timestep between recorded frames
            self.Tgrad = (md_setting[1]-md_setting[0])/(md_setting[3]*md_setting[2]/1000)   # temeperature gradient
            
        
        elif self.data_format == 'npz': # only for internal use, not generalised 
            from pdyna.io import process_lat, chemical_from_formula
            from ase.io.lammpsdata import read_lammps_data as rld
            import pymatgen.io.ase as pia
            
            if len(self.data_path) != 3:
                raise TypeError("The input format for lammps must be (npz_path, MD setting tuple). ")
            npz_path,init_path,lammps_setting = self.data_path 
            print("------------------------------------------------------------")
            print("Loading Trajectory files...")
            
            
            if type(npz_path) is str:
                #specorder = [1,82,6,35,7]
                ll = np.load(npz_path)
                carts = ll['positions']
                cellmat = ll['cells']
                atomic_numbers = ll['numbers']
                
# =============================================================================
#                 ca = []
#                 an = []
#                 for spe in specorder:
#                     ca.append(carts[:,list(np.where(atomic_numbers==spe)[0]),:])
#                     an.append(atomic_numbers[list(np.where(atomic_numbers==spe)[0])])
#                 carts = np.concatenate(ca,axis=1)
#                 atomic_numbers = np.concatenate(an,axis=0)
# =============================================================================
                    
                
                if cellmat.ndim == 2:
                    newmat = np.zeros((cellmat.shape[0],3,3))
                    for i in range(3):
                        newmat[:,i,i]=cellmat[:,i]
                    cellmat = newmat.copy()
                
                lattice = np.empty((0,6))
                for i in range(cellmat.shape[0]):
                    lattice = np.vstack((lattice,process_lat(cellmat[i,:])))
                
                at = rld(init_path, Z_of_type=None, style='atomic', sort_by_id=True, units='metal')
                at.numbers = list(atomic_numbers)
                st0 = pia.AseAtomsAdaptor.get_structure(at)
                
            elif type(npz_path) is list:
                carts = []
                cellmat = []
                for fp in npz_path:
                    ll = np.load(fp)
                    carts.append(ll['positions'])
                    cellmat.append(ll['cells'])
                    atomic_numbers = ll['numbers']
                carts = np.concatenate(carts,axis=0)
                cellmat = np.concatenate(cellmat,axis=0)
                
                lattice = np.empty((0,6))
                for i in range(cellmat.shape[0]):
                    lattice = np.vstack((lattice,process_lat(cellmat[i,:])))
                
                at = rld(init_path, Z_of_type=None, style='atomic', sort_by_id=True, units='metal')
                at.numbers = list(atomic_numbers[0,:])
                st0 = pia.AseAtomsAdaptor.get_structure(at)
            else:
                raise TypeError("The input npz_path must be either str or list. ")
            
            
            
            
            
            for elem in st0.symbol_set:
                if not elem in self._known_elem:
                    raise ValueError(f"An unexpected element {elem} is found. Please check the list known_elem. ")
            
            self.st0 = st0
            self.natom = len(st0)
            self.species_set = st0.symbol_set
            self.formula = chemical_from_formula(st0)
            
            atomic_symbols = []
            for site in st0.sites:
                atomic_symbols.append(site.species_string)
            
            Xindex = []
            Bindex = []
            Cindex = []
            Nindex = []
            Hindex = []
            for i,site in enumerate(atomic_symbols):
                 if site in self._Xsite_species:
                     Xindex.append(i)
                 if site in self._Bsite_species:
                     Bindex.append(i)  
                 if site == 'C':
                     Cindex.append(i)  
                 if site == 'N':
                     Nindex.append(i)  
                 if site == 'H':
                     Hindex.append(i)  
            
            self.Bindex = Bindex
            self.Xindex = Xindex
            self.Cindex = Cindex
            self.Hindex = Hindex
            self.Nindex = Nindex
            
            self.Allpos = carts
            
            self.lattice = lattice
            self.latmat = cellmat
            
            self.nframe = cellmat.shape[0]
            
            self.MDsetting = {}
            self.MDsetting["nsw"] = carts.shape[0]*lammps_setting[3]
            self.MDsetting["Ti"] = lammps_setting[0]
            self.MDsetting["Tf"] = lammps_setting[1]
            self.MDsetting["tstep"] = lammps_setting[2]
            self.MDsetting["nblock"] = lammps_setting[3]
            self.MDTimestep = lammps_setting[2]/1000*lammps_setting[3]  # the timestep between recorded frames
            self.Tgrad = (lammps_setting[1]-lammps_setting[0])/(lammps_setting[3]*lammps_setting[2]/1000)   # temeperature gradient
            
        
        
        
        
        else:
            raise TypeError("Unsupported data format: {}".format(self.data_format))
        
        
        # pre-definitions of the trajectory
        if 'C' in st0.symbol_set:
            self._flag_organic_A = True
        else:
            self._flag_organic_A = False
            
        et1 = time.time()
        self.timing = {}
        self.timing["reading"] = et1-et0
        
    
    def __str__(self):
        pattern = '''
        Perovskite Trajectory
        Formula:          {}
        Number of atoms:  {}
        Number of frames: {}
        Temperature:      {}
        '''
        if self.MDsetting['Ti'] == self.MDsetting['Tf']:
            tstr = str(self.MDsetting['Ti'])+"K"
        else:
            tstr = str(self.MDsetting['Ti'])+"K - "+str(self.MDsetting['Tf'])+"K"
        
        return pattern.format(self.formula, self.natom, self.nframe, tstr)
    
    
    def __repr__(self):
        if self.MDsetting['Ti'] == self.MDsetting['Tf']:
            tstr = str(self.MDsetting['Ti'])+"K"
        else:
            tstr = str(self.MDsetting['Ti'])+"K - "+str(self.MDsetting['Tf'])+"K" 
        return 'PDynA Trajectory({}, {} atoms, {} frames, {})'.format(self.formula, self.natom, self.nframe, tstr)
    
    
    def dynamics(self,
                 # general parameters
                 read_mode: int, # key parameter, 1: static mode, 2: transient mode
                 uniname = "test", # A unique user-defined name for this trajectory, will be used in printing and figure saving
                 allow_equil = 0.5, # take the first x fraction of the trajectory as equilibration, this part will not be computed
                 read_every = 0, # read only every n steps, default is 0 which the code will decide an appropriate value according to the system size
                 saveFigures = False, # whether to save produced figures
                 lib_saver = False,  # whether to save computed material properties in lib file
                 lib_overwrite = False, # whether to overwrite existing lib entry, or just change upon them
                 
                 # function toggles
                 preset = 2, # 0: no preset, uses the toggles, 1: lat & tilt_distort, 2: lat & tilt_distort & tavg & MO, 3: all
                 toggle_lat = False, # switch of lattice parameter calculation
                 toggle_tavg = False, # switch of time averaged structure
                 toggle_tilt_distort = False, # switch of octahedral tilting and distortion calculation
                 toggle_MO = False, # switch of molecular orientation (MO) calculation (for organic A-site)
                 toggle_RDF = False, # switch of radial distribution function calculation
                 toggle_site_disp = False, # switch of A-site cation displacement calculation
                 
                 smoother = 0, # whether to use S-G smoothing on outputs, 0: disabled, >0: average window in ps
                 
                 # Lattice parameter calculation
                 lat_method = 1, # lattice parameter analysis methods, 1: direct lattice cell dimension, 2: pseudo-cubic lattice parameter
                 zdir = 2, # specified z-direction in case of lat_method 2
                 leading_crop = 0.01, # remove the first x fraction of the trajectory on plotting 
                 vis3D_lat = 0, # 3D visualization of lattice parameter in time.
                 
                 # time averaged structure
                 start_ratio = 0.5, # time-averaging structure ratio, e.g. 0.9 means only averaging the last 10% of trajectory
                 tavg_save_dir = ".", # directory for saving the time-averaging structures
                 Asite_reconstruct = False, # setting a different time-averaging algo for organic A-sites
                 
                 # octahedral tilting and distortion
                 structure_type = 1, # 1: 3C polytype, 2: other non-perovskite with orthogonal reference enabled, 3: other non-perovskite with initial config as reference. Please note that mode 2 and 3 are relatively less tested.   
                 multi_thread = 1, # if >1, enable multi-threading in this calculation, since not vectorized
                 rotation_from_orthogonal = None, # None: code will detect if the BX6 frame is not orthogonal to the principle directions, only manually input this [x,y,z] rotation angles in degrees if told by the code. 
                 tilt_corr_NN1 = True, # enable first NN correlation of tilting, reflecting the Glazer notation
                 full_NN1_corr = False, # include off-diagonal correlation terms 
                 tilt_corr_spatial = False, # enable spatial correlation beyond NN1
                 tiltautoCorr = False, # compute Tilting decorrelation time constant
                 octa_locality = False, # compute differentiated properties within mixed-halide sample
                 enable_refit = False, # refit the octahedral network in case of change of geometry
                 symm_n_fold = 0, # tilting range, 0: auto, 2: [-90,90], 4: [-45,45], 8: [0,45]
                 tilt_recenter = False, # whether to eliminate the shift in tilting values according to the mean value of population
                 tilt_domain = False, # compute the time constant of tilt correlation domain formation
                 vis3D_domain = 0, # 3D visualization of tilt domain in time. 0: off, 1: apparent tilting, 2: tilting correlation status
                 
                 # molecular orientation (MO)
                 MOautoCorr = False, # compute MO reorientation time constant
                 MO_corr_spatial = False, # enable spatial correlation function of MO
                 draw_MO_anime = False, # plot the MO in 3D animation, will take a few minutes
                 ):
        
        """
        Core function for analysing perovskite trajectory.
        The parameters are used to enable various analysis functions and handle their functionality.

        Parameters
        ----------
        
        -- General Parameters
        read_mode   -- key parameter, define the reading mode
            1: static mode, treats all the frame equally and output equilibrated properties, generally used for constant-T MD.
            2: transient mode, observes transient properties with respect to changing time or temperature.
        uniname     -- unique user-defined name for this trajectory, will be used in printing and figure saving
        allow_equil -- take the first x (0 to 1) fraction of the trajectory as equilibration, this part will not be computed, used for read_mode 1.
            takes value from 0 to 1, e.g. 0.8 means that only the last 20% of the trajectory will be used for property calculation. default: 0.5
        read_every  -- read only every n steps, default is 0 which the code will decide an appropriate value according to the system size
        
        -- Saving the Outputs
        saveFigures   -- whether to save produced figures
            True or False
        lib_saver     -- whether to save computed material properties in lib file, a lib file written in pickle format will be created if the folder does not contain this file
            True or False
        lib_overwrite -- whether to overwrite existing lib entry, or just change upon them
            True: overwrite the existing entry with properties calculated in this run
            False: properties that are not calculated in this run will be preserved 
        
        -- Function Toggles
        preset              -- Presets of useful function toggles. Overwrtie the individual function toggles if != 0.
            0: no preset, need to manually assign toggles, 
            1: lat & tilt_distort, 
            2: lat & tilt_distort & tavg & MO, 
            3: all functions below
        toggle_lat          -- switch of lattice parameter calculation
            True or False
        toggle_tavg         -- switch of time averaged structure
            True or False
        toggle_tilt_distort -- switch of octahedral tilting and distortion calculation
            True or False
        toggle_MO           -- switch of molecular orientation (MO) calculation (for organic A-site)
            True or False
        toggle_RDF          -- switch of radial distribution function calculation
            True or False
        toggle_site_disp       -- switch of A-site cation displacement calculation
            True or False
        smoother            -- whether to use S-G smoothing on transient outputs, used with read_mode 2
            0: disabled, 
            >0: average time window in ps
        
        -- Lattice Parameter Calculation
        lat_method   -- lattice parameter analysis methods.
            1: direct lattice cell dimension
            2: pseudo-cubic lattice parameter
        zdir         -- specify z-direction in case of lat_method 2.
            0: x, 1: y, 2: z, considering the sample axis, default is 2. 
        leading_crop -- remove the first x (0 to 1) fraction of the trajectory on plotting lattice parameter
            takes value from 0 to 1, e.g. 0.05 means that the first 5% of the trajectory will not be shown. default: 0.01
        vis3D_lat    -- 3D visualization of lattice parameter.
            True or False
        
        -- Time Averaged Structure
        start_ratio       -- the portion of trajectory that will be used for computing time-averaging structure.
            takes value from 0 to 1, e.g. 0.8 means that only the last 20% of the trajectory will be used for computing time-averaging structure. default: 0.5
        tavg_save_dir     -- directory for saving the time-averaging structures, default: current working directory
        Asite_reconstruct -- setting a different time-averaging algo for organic A-sites, only works with MOautoCorr enabled
            True or False
        
        -- Octahedral Tilting and Distortion
        structure_type    -- define connectivity of the perovskite. 
            1: 3C polytype, or equally conner-sharing, default
            2: other non-3C perovskite with orthogonal reference enabled
            3: other non-3C perovskite with initial config as tilting reference     
        multi_thread      -- Enable multi-threading in tilting/distortion calculation, the scaling is near-linear
            1: no multi-threading , default
            >1, enable multi-threading with n threads
        tilt_corr_NN1     -- enable first NN correlation of tilting, reflecting the Glazer notation (a key functionality)
            True or False, default is True
        full_NN1_corr     -- include off-diagonal NN1 correlation terms 
            True or False
        tilt_corr_spatial -- enable spatial correlation beyond NN1
            True or False
        tiltautoCorr      -- compute time-dependent self-correlation of tilting
            True or False
        octa_locality     -- compute differentiated properties within mixed-halide sample
            True or False
        enable_refit      -- refit the octahedral connectivity in case of change of geometry, use with care
            True or False
        symm_n_fold       -- tilting angle symmetry range
            0: auto,  default
            2: [-90,90], 
            4: [-45,45], 
            8: [0,45]
        tilt_recenter     -- whether to eliminate the shift in tilting values according to the mean value of population
            True or False
        tilt_domain       -- compute the time constant of tilt correlation domain formation
            True or False
        vis3D_domain      -- 3D visualization of tilt domain in time. 
            0: off 
            1: apparent tilt angles, 
            2: tilting correlation polarity (TCP)
            
        -- Molecular Orientation (MO)
        MOautoCorr      -- compute MO reorientation time constant
            True or False
        MO_corr_spatial -- enable spatial correlation function of MO
            True or False
        draw_MO_anime   -- plot the MO in 3D animation, will take a few minutes
            True or False
        
        
        p.s. The 'True or False' options all have False as the default unless specified otherwise. 
        """
        
        # pre-definitions
        print("Current sample:",uniname)
        print("Time Span:",round(self.nframe*self.MDTimestep,3),"ps")
        print("Frame count:",self.nframe)
        
        #print("Initializing trajectory")
        
        # reset timing
        self.timing = {"reading": self.timing["reading"]}
        self.uniname = uniname
        
        et0 = time.time()
        if preset == 1:
            toggle_lat = True
            toggle_tavg = False
            toggle_tilt_distort = True
            toggle_MO = False
            toggle_RDF = False
            toggle_site_disp = False
        elif preset == 2:
            toggle_lat = True
            toggle_tavg = True
            toggle_tilt_distort = True
            toggle_MO = True
            toggle_RDF = False
            toggle_site_disp = False
        elif preset == 3:
            toggle_lat = True
            toggle_tavg = True
            toggle_tilt_distort = True
            toggle_MO = True
            toggle_RDF = True
            toggle_site_disp = True
        elif preset == 0:
            pass
        else:
            raise TypeError("The calculation mode preset must be within (0,1,2,3). ")
        
        if multi_thread > 1:
            if enable_refit:
                enable_refit = False
                print("Warning: the refit of octahedral network is disabled due to multi-threading! ")
        
        if not self._flag_organic_A:
            toggle_MO = False # force MO off when no organic components
        
        if not read_mode in [1,2]:
            raise TypeError("Parameter read_mode must be either 1 or 2. ")
        
        if read_mode == 1:
            title = "T="+str(self.MDsetting["Ti"])+"K"
        else:
            #title = stfor+"  T="+str(Ti)+"K-"+str(Tf)+"K"
            title = None
        
        if allow_equil >= 1:
            raise TypeError("Parameter allow_equil must be within 0 to 1 (excluded) or auto (-1). ")
        elif allow_equil < 0:
            auto_equil = 50 # allow 50 ps equilibration
            allow_equil = auto_equil/(self.MDTimestep*self.nframe)
            if allow_equil > 0.95:
                raise TypeError(f"The trajectory is shorter than the equilibration time given ({auto_equil} ps). ")
        
        if read_mode == 2:
            allow_equil = 0
        
        self.allow_equil = allow_equil
        
        if symm_n_fold == 0:
            if structure_type == 2:
                symm_n_fold = 2
            else:
                symm_n_fold = 4
        
        if structure_type in (2,3):
            tilt_corr_NN1 = False
            tilt_corr_spatial = False
            MO_corr_spatial = False
        
        if structure_type in (1,2):
            orthogonal_frame = True
        elif structure_type == 3:
            orthogonal_frame = False
        else:
            raise TypeError("The parameter structure_type must be 1, 2 or 3. ")
        
        if smoother != 0 and read_mode == 2 and self.nframe*self.MDTimestep*0.5 < smoother:
            raise ValueError("The time window for averaging is too large. ")
        
        #if self._flag_cubic_cell == False:
        #    lat_method = 1
        #    tilt_corr_NN1 = False
        #    tilt_domain = False
        
        if self.natom < 200:
            MO_corr_spatial = False 
            tilt_corr_spatial = False 
            tilt_domain = False
            
        if read_every == 0:
            if self.natom < 400:
                read_every = 1
            else:
                if read_mode == 1:
                    read_every = round(self.nframe*(3.6e-05*(len(self.Bindex)**0.7))*(1-allow_equil))
                elif read_mode == 2:
                    read_every = round(self.nframe*(7.2e-05*(len(self.Bindex)**0.7)))
                if read_every < 1:
                    read_every = 1
        
        print(f"Reading every {read_every} frame(s)")
        print(f"Number of atoms: {len(self.st0)}")
        
        if self.MDsetting["Ti"] == self.MDsetting["Ti"]:
            print("Temperature: "+str(self.MDsetting["Ti"])+"K")
        else:
            print("Temperature: "+str(self.MDsetting["Ti"])+"K-"+str(self.MDsetting["Ti"])+"K")
        
        print(" ")
        
        if not tilt_corr_NN1: 
            tilt_domain = False
        
        hal_count = 0
        for sp in self.st0.symbol_set:
            if sp in self._Xsite_species:
                hal_count += 1
        if hal_count == 0:
            raise TypeError("The structure does not contain any recognised X site.")
        elif hal_count == 1:
            octa_locality = False

        
        # end of parameter checking
        self.read_every = read_every
        self.Timestep = self.MDTimestep*read_every # timestep with read_every parameter skipping some steps regularly
        self.timespan = self.nframe*self.Timestep
        self._rotmat_from_orthogonal = None
        
        st0 = self.st0
        at0 = aaa.get_atoms(st0)
        self.at0 = at0
        st0Bpos = st0.cart_coords[self.Bindex,:]
        st0Xpos = st0.cart_coords[self.Xindex,:]
        mymat = st0.lattice.matrix
        
        from pdyna.structural import find_population_gap, apply_pbc_cart_vecs_single_frame
        cell_lat = st0.lattice.abc
        angles = st0.lattice.angles
        if (max(angles) < 100 and min(angles) > 80):
            r0=distance_matrix_handler(st0Bpos,st0Bpos,mymat)
        else:
            r0=distance_matrix_handler(st0Bpos,st0Bpos,None,at0.cell,at0.pbc,True)

        search_NN1 = find_population_gap(r0, self._fpg_val_BB[0], self._fpg_val_BB[1])
        default_BB_dist = np.mean(r0[np.logical_and(r0>0.1,r0<search_NN1)])
        self.default_BB_dist = default_BB_dist
        
        #default_BB_dist = 6.1
        Bpos = self.Allpos[:,self.Bindex,:]
        
        if (max(angles) < 100 and min(angles) > 80):
            ri=distance_matrix_handler(Bpos[0,:],Bpos[0,:],self.latmat[0,:])
            rf=distance_matrix_handler(Bpos[-1,:],Bpos[-1,:],self.latmat[-1,:])
        else:
            ri=distance_matrix_handler(Bpos[0,:],Bpos[0,:],self.latmat[0,:],at0.cell,at0.pbc,True,True)
            rf=distance_matrix_handler(Bpos[-1,:],Bpos[-1,:],self.latmat[-1,:],at0.cell,at0.pbc,True,True)

        if np.amax(np.abs(ri-rf)) > 3: # confirm that no change in the Pb framework
            print("!Tilt-spatial: The difference between the initial and final distance matrix is above threshold ({} A), check ri and rf. \n".format(round(np.amax(np.abs(ri-rf)),3)))
        
        Benv = []
        for B1, B2_list in enumerate(r0): # find the nearest Pb within a cutoff
            Benv.append([i for i,B2 in enumerate(B2_list) if (B2 > 0.1 and B2 < search_NN1)])
        Benv = np.array(Benv)
        
        try:
            aa = Benv.shape[1] # if some of the rows in Benv don't have 6 neighbours.
        except IndexError:
            print(f"Need to adjust the range of B atom 1st NN distance (was {search_NN1}).  ")
            print("See the gap between the populations. \n")
            test_range = ri.reshape((1,ri.shape[0]**2))
            import matplotlib.pyplot as plt
            fig,ax = plt.subplots(1,1)
            plt.hist(test_range.reshape(-1,),range=[5.3,9.0],bins=100)
            #ax.scatter(test_range,test_range)
            #ax.set_xlim([5,10])
        
        self._Benv = Benv
            
        if Benv.shape[1] == 6:
            Bcoordenv = np.empty((Benv.shape[0],6,3))
            for i in range(Benv.shape[0]):
                #Bcoordenv[i,:] = Bpos[0,Benv[i,:],:] - Bpos[0,i,:]
                Bcoordenv[i,:] = st0Bpos[Benv[i,:],:] - st0Bpos[i,:]
            
            Bcoordenv = apply_pbc_cart_vecs_single_frame(Bcoordenv,mymat)
            self._Bcoordenv = Bcoordenv
            
            if rotation_from_orthogonal is None:
                bce = Bcoordenv/default_BB_dist
                csum = np.zeros((6,3))
                for i in range(bce.shape[0]):
                    orders = np.zeros((6,))
                    for j in range(6):
                        orders[j] = np.argmax(np.dot(bce[i,:,:],bce[0,j,:]))
                    csum = csum+bce[i,list(orders.astype(int)),:]
                csum = csum/bce.shape[0]

                orth = np.abs(csum)
                orth[orth>1] = 1-(orth[orth>1]-1)
                if_orth = np.amax(np.amin(np.concatenate((np.abs(orth-1)[:,:,np.newaxis],orth[:,:,np.newaxis]),axis=2),axis=2))
                
                self._non_orthogonal = False
                if if_orth > 0.15 and structure_type == 1: # key modification
                    self._non_orthogonal = True
                    
                if self._non_orthogonal:
                    self._flag_cubic_cell = False
                    self.complex_pbc = True
                    from pdyna.structural import calc_rotation_from_arbitrary_order
                    rots, rtr = calc_rotation_from_arbitrary_order(csum)
                    print(f"Non-orthogonal structure detected, the rotation from the orthogonal reference is: {np.round(rots,3)} degree.")
                    print("If you find this detected rotation incorrect, please use the rotation_from_orthogonal parameter to overwrite this. ")
                    self._rotmat_from_orthogonal = rtr
                    
                else: # is orthogonal frame
                    self._rotmat_from_orthogonal = None
                    angles = self.st0.lattice.angles
                    sides = self.st0.lattice.abc
                    if (max(angles) < 100 and min(angles) > 80):
                        self.complex_pbc = False
                        if  (max(sides)-min(sides))/6 < 0.8:
                            self._flag_cubic_cell = True
                        else:
                            self._flag_cubic_cell = False
                    else:
                        self._flag_cubic_cell = False  
                        self.complex_pbc = True
            else: # manually input a rotation from reference
                self._non_orthogonal = True
                self._flag_cubic_cell = False 
                self.complex_pbc = True
                from scipy.spatial.transform import Rotation as sstr
                tempvec = np.array(rotation_from_orthogonal)/180*np.pi
                rtr = sstr.from_rotvec(tempvec).as_matrix().reshape(3,3)
                self._rotmat_from_orthogonal = rtr
                
            
        elif Benv.shape[1] == 3:
            self._flag_cubic_cell = True
            self._non_orthogonal = False
            self.complex_pbc = False
                   
        else:
            self._flag_cubic_cell = False
            self._non_orthogonal = False
            self.complex_pbc = True
            print(f"The B-B environment matrix is {Benv.shape[1]}. This indicates a non-3C polytype (3 or 6). ")
        
            #raise TypeError(f"The environment matrix is incorrect. The connectivity is {Benv.shape[1]} instead of 6. ")
        
        if self._flag_cubic_cell:
            supercell_size = round(np.mean(cell_lat)/default_BB_dist)
            if abs(supercell_size - (len(self.Bindex))**(1/3)) > 0.0001:
                raise ValueError("The computed supercell size does not match with the number of octahedra.")
            self.supercell_size = supercell_size
        
        # label the constituent octahedra
        if toggle_tavg or toggle_tilt_distort or toggle_site_disp: 
            from pdyna.structural import fit_octahedral_network_defect_tol, fit_octahedral_network_defect_tol_non_orthogonal, find_polytype_network
            rt = distance_matrix_handler(st0Bpos,st0Xpos,mymat,self.at0.cell,self.at0.pbc,self.complex_pbc)
            
            if orthogonal_frame:
                if not self._non_orthogonal:
                    neigh_list = fit_octahedral_network_defect_tol(st0Bpos,st0Xpos,rt,mymat,self._fpg_val_BX,structure_type)
                else: # non-orthogonal
                    neigh_list = fit_octahedral_network_defect_tol_non_orthogonal(st0Bpos,st0Xpos,rt,mymat,self._fpg_val_BX,structure_type,self._rotmat_from_orthogonal)
                self.octahedra = neigh_list
            else:
                neigh_list, ref_initial = fit_octahedral_network_defect_tol(st0Bpos,st0Xpos,rt,mymat,self._fpg_val_BX,structure_type)
                self.octahedra = neigh_list
                self.octahedra_ref = ref_initial
            
            # determine polytype (experimental)
            #rt=distance_matrix_handler(st0Bpos,st0Xpos,mymat,self.at0.cell,self.at0.pbc,self.complex_pbc)
            #conntypeStr, connectivity, conn_category = find_polytype_network(st0Bpos,st0Xpos,rt,mymat,neigh_list)
            #self.octahedral_connectivity = conn_category
        
        # label the constituent A-sites
        if toggle_MO or toggle_site_disp:
            from pdyna.io import display_A_sites
            
            Nindex = self.Nindex
            Cindex = self.Cindex
            
            # recognise all organic molecules
            Aindex_fa = []
            Aindex_ma = []
            Aindex_azr = []
            
            dm = distance_matrix_handler(st0.cart_coords[Cindex,:],st0.cart_coords[Nindex,:],mymat,at0.cell,at0.pbc,self.complex_pbc)
            dm1 = distance_matrix_handler(st0.cart_coords[Cindex,:],st0.cart_coords[Cindex,:],mymat,at0.cell,at0.pbc,self.complex_pbc)
            
            CN_max_distance = 2.5
            
            # search all A-site cations and their constituent atoms (if organic)
            Cpass = []
            for i in range(dm.shape[0]):
                if i in Cpass: continue # repetitive C atom in case of Azr
                
                Ns = []
                temp = np.argwhere(dm[i,:] < CN_max_distance).reshape(-1)
                for j in temp:
                    Ns.append(Nindex[j])
                moreC = np.argwhere(np.logical_and(dm1[i,:]<CN_max_distance,dm1[i,:]>0.01)).reshape(-1)
                if len(moreC) == 1: # aziridinium
                    Aindex_azr.append([[Cindex[i],Cindex[moreC[0]]],Ns])
                    Cpass.append(moreC[0])
                elif len(moreC) == 0:
                    if len(temp) == 1:
                        Aindex_ma.append([[Cindex[i]],Ns])
                    elif len(temp) == 2:
                        Aindex_fa.append([[Cindex[i]],Ns])
                    else:
                        raise ValueError(f"There are {len(temp)} N atom connected to C atom number {i}")
                else:
                    raise ValueError(f"There are {len(moreC)+1} C atom connected to C atom number {i}")
                    
            Aindex_cs = []
            
            for i,site in enumerate(st0.sites):
                 if site.species_string == 'Cs':
                     Aindex_cs.append(i)  
            
            self.A_sites = {"FA": Aindex_fa, "MA": Aindex_ma, "Cs": Aindex_cs, "Azr": Aindex_azr }
            display_A_sites(self.A_sites)
        
        et1 = time.time()
        self.timing["env_resolve"] = et1-et0
        
        # use a lib file to store computed dynamic properties
        self.prop_lib = {}
        self.prop_lib['Ti'] = self.MDsetting['Ti']
        self.prop_lib['Tf'] = self.MDsetting['Tf']
        
        
        # running calculations
        if toggle_lat:
            self.lattice_parameter(lat_method=lat_method,uniname=uniname,read_mode=read_mode,allow_equil=allow_equil,zdir=zdir,smoother=smoother,leading_crop=leading_crop,saveFigures=saveFigures,title=title,vis3D_lat=vis3D_lat)
        
        if toggle_tilt_distort:
            print("Computing octahedral tilting and distortion...")
            self.tilting_and_distortion(uniname=uniname,multi_thread=multi_thread,read_mode=read_mode,read_every=read_every,allow_equil=allow_equil,tilt_corr_NN1=tilt_corr_NN1,tilt_corr_spatial=tilt_corr_spatial,enable_refit=enable_refit,symm_n_fold=symm_n_fold,saveFigures=saveFigures,smoother=smoother,title=title,orthogonal_frame=orthogonal_frame,structure_type=structure_type,tilt_domain=tilt_domain,vis3D_domain=vis3D_domain,tilt_recenter=tilt_recenter,full_NN1_corr=full_NN1_corr,tiltautoCorr=tiltautoCorr)
            if read_mode == 1:
                print("dynamic distortion:",np.round(self.prop_lib["distortion"][0],4))
            if structure_type in (1,3) and read_mode == 1:
                print("dynamic tilting:",np.round(self.prop_lib["tilting"].reshape(3,),3))
            if 'tilt_corr_polarity' in self.prop_lib and read_mode == 1:
                print("tilting correlation:",np.round(np.array(self.prop_lib['tilt_corr_polarity']).reshape(3,),3))
            print(" ")
            
        if toggle_MO:
            self.molecular_orientation(uniname=uniname,read_mode=read_mode,allow_equil=allow_equil,MOautoCorr=MOautoCorr, MO_corr_spatial=MO_corr_spatial, title=title,saveFigures=saveFigures,smoother=smoother,draw_MO_anime=draw_MO_anime)    
        
        if toggle_tavg:
            et0 = time.time()
            from pdyna.structural import structure_time_average_ase, structure_time_average_ase_organic, simply_calc_distortion
            
            if (not Asite_reconstruct): # or (not 'reorientation' in self.prop_lib):
                #if Asite_reconstruct and (not 'reorientation' in self.prop_lib):
                #    print("!Time-averaged structure: please enable MOautoCorr to allow Asite_reconstruct. ")
                struct = structure_time_average_ase(self,start_ratio= start_ratio, cif_save_path=tavg_save_dir+f"\\{uniname}_tavg.cif")
            else: # Asite_reconstruct is on and reorientation has been calculated
                if ('reorientation' in self.prop_lib):
                    dec = []
                    for elem in self.prop_lib['reorientation']:
                        dec1 = self.prop_lib['reorientation'][elem]
                        if type(dec1) == int:
                            dec.append(dec1)
                        elif type(dec1) == list:
                            dec.append(sum(dec1)/len(dec1))
                    tavgspan = round(min(dec)/self.MDTimestep/3)
                else: # MO_autocorr is not calculated
                    tavgspan = round(5/self.MDTimestep/3)
                    if tavgspan < 1: tavgspan = 1
                
                struct = structure_time_average_ase_organic(self,tavgspan,start_ratio= start_ratio, cif_save_path=tavg_save_dir+f"\\{uniname}_tavg.cif")

            self.tavg_struct = struct
            tavg_dist = simply_calc_distortion(self)[0]
            print("time-averaged structure distortion mode: ")
            print(np.round(tavg_dist,4))
            print(" ")

            self.prop_lib['distortion_tavg'] = tavg_dist
            
            et1 = time.time()
            self.timing["tavg"] = et1-et0
        
        if toggle_RDF:
            self.radial_distribution(allow_equil=allow_equil,uniname=uniname,saveFigures=saveFigures)
        
        if toggle_site_disp:
            self.site_displacement(allow_equil=allow_equil,uniname=uniname,saveFigures=saveFigures)
        
        if octa_locality:
            self.property_processing(allow_equil=allow_equil,read_mode=read_mode,octa_locality=octa_locality,uniname=uniname,saveFigures=saveFigures)
        
        if read_mode == 2 and toggle_lat and toggle_lat and toggle_tilt_distort and toggle_MO and tilt_corr_NN1 and MO_corr_spatial:
            from pdyna.analysis import draw_transient_properties
            draw_transient_properties(self.Lobj, self.Tobj, self.Cobj, self.Mobj, uniname, saveFigures)
        
        if lib_saver and read_mode == 1:
            import pickle
            lib_name = "perovskite_gaussian_data"
            if not os.path.isfile(lib_name): # create the data file if not found in work dir
                datalib = {}
                with open(lib_name, "wb") as fp:   #Pickling
                    pickle.dump(datalib, fp)

            with open(lib_name, "rb") as fp:   # Unpickling
                datalib = pickle.load(fp)
            
            if lib_overwrite:
                datalib[uniname] = self.prop_lib
            else:
                if not uniname in datalib:
                    datalib[uniname] = {}
                for ent in self.prop_lib:
                    datalib[uniname][ent] = self.prop_lib[ent]
            
            with open(lib_name, "wb") as fp:   #Pickling
                pickle.dump(datalib, fp)
                
            print("lib_saver: Calculated properties are saved in the data library. ")
            
        # summary
        self.timing["total"] = sum(list(self.timing.values()))
        print(" ")
        print_time(self.timing)
        
        # end of calculation
        print("------------------------------------------------------------")
    
    
    
    def lattice_parameter(self,uniname,lat_method,read_mode,allow_equil,zdir,smoother,leading_crop,saveFigures,title,vis3D_lat):
        
        """
        Lattice parameter analysis methods.

        Parameters
        ----------
        lat_method : lattice parameter analysis methods.
            1: direct lattice parameters from cell dimension
            2: pseudo-cubic lattice parameters
        allow_equil: take the first x fraction of the trajectory as equilibration
        zdir : specified z-direction in case of lat_method 2
            0 -> a-axis, 1 -> b-axis, 2 -> c-axis
        smoother: whether to use S-G smoothing on outputs 
        leading_crop: remove the first x fraction of the trajectory on plotting 
        """
        
        et0 = time.time()
        
        if lat_method == 1:
            Lat = np.empty((0,3))
            Lat[:] = np.NaN
            
            st0lat = self.st0.lattice
# =============================================================================
#             std_param = [np.std(np.array([st0lat.a,st0lat.b,st0lat.c])),
#                          np.std(np.array([st0lat.a/np.sqrt(2),st0lat.b/np.sqrt(2),st0lat.c/2]))]
# 
#             if std_param.index(min(std_param)) == 1:
#                 temp = self.lattice[round(self.nframe*allow_equil):,:3]
#                 temp[:,:2] = temp[:,:2]/np.sqrt(2)
#                 temp[:,2] = temp[:,2]/2
#                 Lat = temp
#                     
#             elif std_param.index(min(std_param)) == 0:
#                 Lat = self.lattice[round(self.nframe*allow_equil):,:3]
# =============================================================================
            Lat = self.lattice[round(self.nframe*allow_equil):,:3]
            if self._flag_cubic_cell: # if cubic cell
                Lat_scale = round(np.nanmean(Lat[0,:3])/self.default_BB_dist)
                Lat = Lat/Lat_scale
            else:
                print("!lattice_parameter: detected non-cubic cell, use direct cell dimension as output. ")
            
            self.Lat = Lat
              
        elif lat_method == 2:
            
            from pdyna.structural import pseudocubic_lat
            Lat = pseudocubic_lat(self, allow_equil, zdrc=zdir, lattice_tilt=self._rotmat_from_orthogonal)
            self.Lat = Lat
        
        else:
            raise TypeError("The lat_method parameter must be 1 or 2. ")
        
        num_crop = round(self.nframe*(1-self.allow_equil)*leading_crop)      
        
        # data visualization
        if type(Lat) is list:
            for sublat in Lat:
                if read_mode == 1:
                    from pdyna.analysis import draw_lattice_density
                    if self._flag_cubic_cell:
                        Lmu, Lstd = draw_lattice_density(sublat, uniname=uniname,saveFigures=saveFigures, n_bins = 50, num_crop = num_crop,screen = [5,8], title=title) 
                    else:
                        Lmu, Lstd = draw_lattice_density(sublat, uniname=uniname,saveFigures=saveFigures, n_bins = 50, num_crop = num_crop, title=title) 
                    
                else: #quench mode
                    if self.Tgrad != 0: 
                        Ti = self.MDsetting["Ti"]
                        Tf = self.MDsetting["Tf"]
                        if self.nframe*self.MDsetting["nblock"] < self.MDsetting["nsw"]*0.99: # with tolerance
                            print("Lattice: Incomplete run detected! \n")
                            Ti = self.MDsetting["Ti"]
                            Tf = self.MDsetting["Ti"]+(self.MDsetting["Tf"]-self.MDsetting["Ti"])*(self.nframe*self.MDsetting["nblock"]/self.MDsetting["nsw"])
                        if sublat.shape[0] != self.nframe: # read_every is not 1
                            steps1 = np.linspace(Ti,Tf,sublat.shape[0])
                        else:
                            steps1 = np.linspace(Ti,Tf,self.nframe)

                        invert_x = False
                        if Tf<Ti:
                            invert_x = True
                        
                        self.Ltempline = steps1
                        
                        from pdyna.analysis import draw_lattice_evolution
                        draw_lattice_evolution(sublat, steps1, Tgrad = self.Tgrad, uniname=uniname, saveFigures = saveFigures, xaxis_type = 'T', Ti = Ti,invert_x=invert_x) 
             
                    else: 
                        timeline = np.linspace(1,sublat.shape[0],sublat.shape[0])*self.MDTimestep
                        self.Ltimeline = timeline
                        
                        from pdyna.analysis import draw_lattice_evolution_time
                        _ = draw_lattice_evolution_time(sublat, timeline, self.MDsetting["Ti"],uniname = uniname, saveFigures = False, smoother = smoother) 
            
        else: 
            if read_mode == 1:
                from pdyna.analysis import draw_lattice_density
                if self._flag_cubic_cell:
                    Lmu, Lstd = draw_lattice_density(Lat, uniname=uniname,saveFigures=saveFigures, n_bins = 50, num_crop = num_crop,screen = [5,8], title=title) 
                else:
                    Lmu, Lstd = draw_lattice_density(Lat, uniname=uniname,saveFigures=saveFigures, n_bins = 50, num_crop = num_crop, title=title) 
                # update property lib
                self.prop_lib['lattice'] = [Lmu,Lstd]
                
            else: #quench mode
                if self.Tgrad != 0: 
                    Ti = self.MDsetting["Ti"]
                    Tf = self.MDsetting["Tf"]
                    if self.nframe*self.MDsetting["nblock"] < self.MDsetting["nsw"]*0.99: # with tolerance
                        print("Lattice: Incomplete run detected! \n")
                        Ti = self.MDsetting["Ti"]
                        Tf = self.MDsetting["Ti"]+(self.MDsetting["Tf"]-self.MDsetting["Ti"])*(self.nframe*self.MDsetting["nblock"]/self.MDsetting["nsw"])
                    if Lat.shape[0] != self.nframe: # read_every is not 1
                        steps1 = np.linspace(Ti,Tf,Lat.shape[0])
                    else:
                        steps1 = np.linspace(Ti,Tf,self.nframe)

                    invert_x = False
                    if Tf<Ti:
                        invert_x = True
                        
                    self.Ltempline = steps1
                    from pdyna.analysis import draw_lattice_evolution
                    draw_lattice_evolution(Lat, steps1, Tgrad = self.Tgrad, uniname=uniname, saveFigures = saveFigures, xaxis_type = 'T', Ti = Ti,invert_x=invert_x) 
         
                else: 
                    timeline = np.linspace(1,Lat.shape[0],Lat.shape[0])*self.MDTimestep
                    self.Ltimeline = timeline
                    
                    from pdyna.analysis import draw_lattice_evolution_time
                    self.Lobj = draw_lattice_evolution_time(Lat, timeline, self.MDsetting["Ti"],uniname = uniname, saveFigures = False, smoother = smoother) 
            

        if vis3D_lat != 0 and vis3D_lat in (1,2):
            from scipy.stats import binned_statistic_dd as binstat
            from pdyna.analysis import savitzky_golay, vis3D_domain_anime

            axisvis = 2

            cc = self.st0.frac_coords[self.Bindex,:]
            
            supercell_size = self.supercell_size
            
            clims = np.array([[(np.quantile(cc[:,0],1/(supercell_size**2))+np.amin(cc[:,0]))/2,(np.quantile(cc[:,0],1-1/(supercell_size**2))+np.amax(cc[:,0]))/2],
                              [(np.quantile(cc[:,1],1/(supercell_size**2))+np.amin(cc[:,1]))/2,(np.quantile(cc[:,1],1-1/(supercell_size**2))+np.amax(cc[:,1]))/2],
                              [(np.quantile(cc[:,2],1/(supercell_size**2))+np.amin(cc[:,2]))/2,(np.quantile(cc[:,2],1-1/(supercell_size**2))+np.amax(cc[:,2]))/2]])
            
            bin_indices = binstat(cc, None, 'count', bins=[supercell_size,supercell_size,supercell_size], 
                                  range=[[clims[0,0]-0.5*(1/supercell_size), 
                                          clims[0,1]+0.5*(1/supercell_size)], 
                                         [clims[1,0]-0.5*(1/supercell_size), 
                                          clims[1,1]+0.5*(1/supercell_size)],
                                         [clims[2,0]-0.5*(1/supercell_size), 
                                          clims[2,1]+0.5*(1/supercell_size)]],
                                  expand_binnumbers=True).binnumber
            # validate the binning
            atom_indices = np.array([bin_indices[0,i]+(bin_indices[1,i]-1)*supercell_size+(bin_indices[2,i]-1)*supercell_size**2 for i in range(bin_indices.shape[1])])
            bincount = np.unique(atom_indices, return_counts=True)[1]
            if len(bincount) != supercell_size**3:
                raise TypeError("Incorrect number of bins. ")
            if max(bincount) != min(bincount):
                raise ValueError("Not all bins contain exactly the same number of atoms (1). ")
            
            if vis3D_lat == 1:

                l = Lat.copy()
                
                time_window = 5 # picosecond
                sgw = round(time_window/self.Timestep)
                if sgw<5: sgw = 5
                if sgw%2==0: sgw+=1
                
                for i in range(l.shape[1]):
                    for j in range(3):
                        temp = l[:,i,j]
                        temp = savitzky_golay(temp,window_size=sgw)
                        l[:,i,j] = temp

                polfeat = l[:,:,2]-np.amin(l[:,:,:2],axis=2)
                polfeat[polfeat<0] = 0
                polfeat = np.clip(polfeat,0,np.quantile(polfeat,0.90))
                polfeat = polfeat/np.amax(polfeat)
                
                polfeat = np.power(polfeat,1/2)
                
                polfeat[polfeat<0.2] = 0
                
                plotfeat = polfeat.copy()

            elif vis3D_lat == 2:
                
                l = Lat.copy()
                
                time_window = 5 # picosecond
                sgw = round(time_window/self.Timestep)
                if sgw<5: sgw = 5
                if sgw%2==0: sgw+=1
                
                for i in range(l.shape[1]):
                    for j in range(3):
                        temp = l[:,i,j]
                        temp = savitzky_golay(temp,window_size=sgw)
                        l[:,i,j] = temp

                polfeat = np.abs(l[:,:,0]-l[:,:,1])
                polfeat = np.clip(polfeat,0,np.quantile(polfeat,0.90))
                polfeat = polfeat/np.amax(polfeat)
                
                polfeat = np.power(polfeat,1/2)
                
                polfeat[polfeat<0.2] = 0
                
                plotfeat = polfeat.copy()
            
            def map_rgb(x):
                return plt.cm.Blues(x)
            
            cfeat = map_rgb(plotfeat)
            
            
            readtime = 60 # ps
            readfreq = 0.6 # ps
            fstart = round(cfeat.shape[0]*0.1)
            fend = min(fstart+round(readtime/self.MDTimestep),round(cfeat.shape[0]*0.9))
            frs = list(np.round(np.linspace(fstart,fend,round((fend-fstart)*self.MDTimestep/readfreq)+1)).astype(int))
            
            figname1 = f"lat3D_{uniname}"
            et0 = time.time()
            vis3D_domain_anime(cfeat,frs,self.MDTimestep,self.supercell_size,bin_indices,figname1)
            et1 = time.time()
            print(round((et1-et0)/60,2))
        
        if read_mode == 1 and self.Tgrad == 0:
            if lat_method == 1:
                print("Lattice parameter: ",np.round(Lmu,4))
            elif lat_method == 2:
                print("Pseudo-cubic lattice parameter: ",np.round(Lmu,4))
            print("")
            
        et1 = time.time()
        self.timing["lattice"] = et1-et0
        


    def tilting_and_distortion(self,uniname,multi_thread,read_mode,read_every,allow_equil,tilt_corr_NN1,tilt_corr_spatial,enable_refit, symm_n_fold,saveFigures,smoother,title,orthogonal_frame,structure_type,tilt_domain,vis3D_domain,tilt_recenter,full_NN1_corr,tiltautoCorr):
        
        """
        Octhedral tilting and distribution analysis.

        Parameters
        ----------
        multi_thread : number of multi-threading for this calculation, input 1 to disable
        orthogonal_frame : use True only for 3C polytype with octahedral coordination number of 6
        tilt_corr_NN1 : enable first NN correlation of tilting, reflecting the Glazer notation
        tilt_corr_spatial : enable spatial correlation beyond NN1
        enable_refit : refit octahedral network when abnormal distortion values are detected (indicating change of network)
            - only turn on when rearrangement is observed
        symm_n_fold: enable to fold the negative axis of tilting status leaving angle in [0,45] degree

        """
        
        from pdyna.structural import resolve_octahedra
        from pdyna.analysis import compute_tilt_density
        
        et0 = time.time()
        
        st0 = self.st0
        lattice = self.lattice
        latmat = self.latmat
        Bindex = self.Bindex
        Xindex = self.Xindex
        Bpos = self.Allpos[:,self.Bindex,:]
        Xpos = self.Allpos[:,self.Xindex,:]
        neigh_list = self.octahedra
        
        mymat=st0.lattice.matrix
        
        # tilting and distortion calculations
        ranger = self.nframe
        timeline = np.linspace(1,ranger,ranger)*self.MDTimestep
        if allow_equil == 0:
            ranger0 = 0
        elif allow_equil > 0:
            ranger0 = round(ranger*allow_equil)
            timeline = timeline[round(timeline.shape[0]*allow_equil):]
        
        readfr = list(range(ranger0,ranger,self.read_every))
        self.frames_read = readfr
    
        if orthogonal_frame:
            ref_initial = None
        else:
            ref_initial = self.octahedra_ref 
        
        rotation_from_orthogonal = None
        if self._non_orthogonal:
            rotation_from_orthogonal = self._rotmat_from_orthogonal
            Di, T, refits = resolve_octahedra(Bpos,Xpos,readfr,self.at0,enable_refit,multi_thread,lattice,latmat,self._fpg_val_BX,neigh_list,orthogonal_frame,structure_type,self.complex_pbc,ref_initial,np.linalg.inv(rotation_from_orthogonal))
        else:
            Di, T, refits = resolve_octahedra(Bpos,Xpos,readfr,self.at0,enable_refit,multi_thread,lattice,latmat,self._fpg_val_BX,neigh_list,orthogonal_frame,structure_type,self.complex_pbc,ref_initial)
        
        if tilt_recenter:
            recenter = []
            for i in range(3):
                if abs(np.nanmean(T[:,:,i])) > 1:
                    T[:,:,i] = T[:,:,i]-np.nanmean(T[:,:,i])
                    recenter.append(i)
            if len(recenter) > 0:
                print(f"!Tilting: detected shifted tilting values in axes {recenter}, the population is re-centered.")
        
        hasDefect = False
        if np.amax(Di) > 1 and np.amax(np.abs(T)) > 45:
            print(f"!Tilting and Distortion: detected some distortion values ({round(np.amax(Di),3)}) larger than 1 and some tilting values ({round(np.amax(np.abs(T)),1)}) outside the range -45 to 45 degree, consider defect formation. ")
            hasDefect = True
            
        if read_every > 1: # deal with the timeline if skipping some steps
            temp_list = []
            for i,temp in enumerate(timeline):
                if i%read_every == 0:
                    temp_list.append(temp)
            timeline = np.array(temp_list)
            
            if timeline.shape[0] == Di.shape[0]+1:
                timeline = timeline[1:]
            
            assert timeline.shape[0] == Di.shape[0]
            assert timeline.shape[0] == T.shape[0]
        
        if np.sum(refits[:,1]) > 0:
            print(f"!Refit: There are {int(refits.shape[0])} re-fits in the run, and some of them detected changed coordination system. \n")
            print(refits)
        
        # screening
# =============================================================================
#         T[np.abs(T)>45] = np.nan
#         dscreen = [0.3,0.8,0.4,0.8]
#         for i in range(4):
#             Di[Di[:,:,i]>dscreen[i],:] = np.nan
# =============================================================================
        
        self.TDtimeline = timeline
        self.Distortion = Di
        self.Tilting = T
        
        # finding defects if any
# =============================================================================
#         def_tilt = np.sum(np.sum(np.abs(T)>45,axis=2)>0,axis=1)
#         def_dist = np.sum(np.sum(Di>0.7,axis=2)>0,axis=1)
#         tl = self.TDtimeline
#         plt.plot(tl,def_dist,label="Distortion")
#         plt.plot(tl,def_tilt,label="Tilting")
#         plt.xlabel("Time (ps)")
#         plt.ylabel("Number of defected octahedra")
#         plt.legend()
# =============================================================================
        
# =============================================================================
#         dmax = np.amax(Di,axis=2)
#         defect_array = np.array(np.where(dmax>0.8))
#         unique, counts = np.unique(defect_array[1,:], return_counts=True)
#         defs = np.concatenate((unique[:,np.newaxis], counts[:,np.newaxis]),axis=1)
#         def_sites = [defs[a,0] for a in range(defs.shape[0]) if defs[a,1] > Di.shape[0]*0.2]
#         if len(def_sites) > 0:
#             self.defect_octahedra = def_sites
#             if len(def_sites) > Di.shape[1]*0.01:
#                 print(f"!Defecfs: Found {len(def_sites)} octahedra with defects formed. ")
#         else:
#             self.defect_octahedra = None
#         #set(def_sites1).intersection(set(def_sites2))
#         
#         b0 = st0.cart_coords[Bindex,:]
#         x0 = st0.cart_coords[Xindex,:]
#         r0 = distance_matrix_handler(b0,x0,mymat,at0.cell,at0.pbc,self.complex_pbc)
#         b1 = Bpos[-1,:]
#         x1 = Xpos[-1,:]
#         rf = distance_matrix_handler(b1,x1,self.latmat[-1,:],at0.cell,at0.pbc,self.complex_pbc)
#         
#         diffr = np.abs(rf-r0)
#         diffb = []
#         for i in range(diffr.shape[0]):
#             diffb.append(np.mean(diffr[i,r0[i,:] < 11]))
#         diffb = np.array(diffb)
#         np.where(diffb>1.5)
#         diffx = []
#         for i in range(diffr.shape[1]):
#             diffx.append(np.mean(diffr[r0[:,i] < 15,i]))
#         diffx = np.array(diffx)
#         np.where(diffx>1.5)
# =============================================================================
        
        
# =============================================================================
#         from sklearn.decomposition import PCA
#         from sklearn.cluster import KMeans
#         from pdyna.analysis import savitzky_golay
#         
#         timeline1 = self.TDtimeline
#         tspan = 50 # ps
#         trajnum = np.argmin(np.abs(timeline1[-1] - timeline1 - tspan))
#         
#         Davg = np.nanmean(Di[trajnum:,:],axis=0)
#         
#         for ra in np.linspace(1,0.8,21):
#             if ra == 1: ra = 1
#             pca=PCA(n_components=ra) #cumulative contribution: % 
#             Y=pca.fit_transform(Davg)
#             if Y.shape[1] == 2: break
#         if Y.shape[1] != 2:
#             raise ValueError("Unsuccessful PCA: The dimensionality of resulting components is {}, instead of the required value 2. \n".format(Y.shape[1]))
#         
#         kmeans = KMeans(n_clusters=2, random_state=0).fit(Y)
#         np.sum(kmeans.labels_)
#         colors = ['r','g','b','c','m', 'y', 'k']
#         carr = []
#         for i in range(Y.shape[0]):
#             carr.append(colors[int(kmeans.labels_[i])])
#         carr = np.array(carr)
#         
#         kmeans = KMeans(n_clusters=2, random_state=0).fit(Davg)
#         np.sum(kmeans.labels_)
#         colors = ['r','g','b','c','m', 'y', 'k']
#         carr = []
#         for i in range(Y.shape[0]):
#             carr.append(colors[int(kmeans.labels_[i])])
#         carr = np.array(carr)
#         
#         plt.scatter(Y[:,0],Y[:,1],marker="o", color=carr)
#         
#         
#         conn = self.octahedra
#         
#         Cavg = round((trajnum/len(timeline1))*self.Allpos.shape[0])
#         initfrac = self.st0.frac_coords[:,:]
#         newfrac = np.mean(get_frac_from_cart(self.Allpos[Cavg:,:,:],self.latmat[Cavg:,:]),axis=0)
#         newpos = np.mean(self.Allpos[Cavg:,:,:],axis=0)
#         selectpos = (initfrac-newfrac)*self.supercell_size
#         
#         Bdisp = selectpos[self.Bindex,:]
#         Xdisp = selectpos[self.Xindex,:]
#         
#         Bdnorm = np.linalg.norm(Bdisp,axis=1)
#         Xdnorm = np.linalg.norm(Xdisp,axis=1)
#         
#         disp_filt = 0.2
#         if np.amax(Xdnorm) > disp_filt:
#             xwhere = np.where(Xdnorm>disp_filt)
#             xdv = np.empty((0,3))
#             for xi in list(xwhere[0]):
#                 xdv = np.vstack((xdv,newpos[self.Xindex[xi],:] - newpos[self.Bindex[np.where(conn==xi)[0][0]],:]))
#             xdv = xdv/default_BB_dist
#             xdv = np.abs(xdv - np.round(xdv))
#             
#         defect_3D_scatter(xdv,uniname,"X", saveFigures)
#         
#         #887,4093,1021,951
#         bind = 887
#         plt.scatter(Y[bind,0],Y[bind,1],marker="^", color='m',s=50)
#         bd = Bpos[:,[bind],:]
#         xd = Xpos[:,self.octahedra[bind],:]
#         
#         ps = np.concatenate((bd,xd),axis=1)
#         prange = np.array([[np.amax(ps[:,:,0]),np.amax(ps[:,:,1]),np.amax(ps[:,:,2])],
#                            [np.amin(ps[:,:,0]),np.amin(ps[:,:,1]),np.amin(ps[:,:,2])]])
#         pwide = np.amax(prange[0,:]-np.mean(prange,axis=0))
#         plim = np.array([[1,1,1],[-1,-1,-1]])*pwide+np.mean(prange,axis=0)
#         
#         from matplotlib.animation import FuncAnimation, PillowWriter
#         
#         frs = list(range(len(self.TDtimeline)))
#         readfr = self.frames_read
#         
#         fig=plt.figure()
#         ax = plt.axes(projection='3d')
# 
#         def animate(i):
#             ax.clear()
#             timelabel = ax.text2D(0.85, 0.14, f"{round(timeline1[i],1)} ps", ha='center', va='center', fontsize=12, color="k", transform=ax.transAxes)
#             distlabel = ax.text2D(0.60, 0.06, f"D={np.round(Di[i,bind,:],3)}", ha='center', va='center', fontsize=12, color="k", transform=ax.transAxes)
#             
#             scat1 = ax.scatter(bd[readfr[i],[0],0], bd[readfr[i],[0],1], bd[readfr[i],[0],2],color="c",s=50,alpha = 1)
#             scat2 = ax.scatter(xd[readfr[i],:,0], xd[readfr[i],:,1], xd[readfr[i],:,2],color="g",s=35,alpha = 1)
#             ax.set_facecolor("grey")
#             #ax.axis('off')
#             ax.set_aspect('auto')
#             ax.set_xticklabels([])
#             ax.set_yticklabels([])
#             ax.set_zticklabels([])
#             ax.set_xlim([plim[1,0],plim[0,0]])
#             ax.set_ylim([plim[1,1],plim[0,1]])
#             ax.set_zlim([plim[1,2],plim[0,2]])
#             return fig
#         
#         anim = FuncAnimation(fig, animate, frames=frs, interval=500)
#         writer = PillowWriter(fps=15)
#         #anim.save(f"{figname}.gif", writer=writer)
#         anim.save("link-anim.gif", writer=writer)
#         plt.show()
# =============================================================================
        
        
        
        
        
        # data visualization
        if read_mode == 2:
            if self.Tgrad == 0:
                from pdyna.analysis import draw_tilt_evolution_time, draw_tilt_corr_density_time, draw_dist_evolution_time
                self.Tobj = draw_tilt_evolution_time(T, timeline,uniname, saveFigures=False, smoother=smoother)
                self.Dobj = draw_dist_evolution_time(Di, timeline,uniname, saveFigures=False, smoother=smoother)
            
            else:
                from pdyna.analysis import draw_dist_evolution, draw_tilt_evolution
                Ti = self.MDsetting["Ti"]
                Tf = self.MDsetting["Tf"]
                if self.nframe*self.MDsetting["nblock"] < self.MDsetting["nsw"]*0.99: # with tolerance
                    print("Tilt & Dist: Incomplete MD run detected! \n")
                    Ti = self.MDsetting["Ti"]
                    Tf = self.MDsetting["Ti"]+(self.MDsetting["Tf"]-self.MDsetting["Ti"])*(self.nframe*self.MDsetting["nblock"]/self.MDsetting["nsw"])
                steps = np.linspace(self.MDsetting["Ti"],self.MDsetting["Tf"],self.nframe)
                
                if read_every != 0:
                    temp_list = []
                    for i,temp in enumerate(steps):
                        if i%read_every == 0:
                            temp_list.append(temp)
                    steps = np.array(temp_list)
                    
                    assert steps.shape[0] == Di.shape[0]
                    assert steps.shape[0] == T.shape[0]

                invert_x = False
                if Tf<Ti:
                    invert_x = True
                
                self.TDtempline = steps
                #draw_distortion_evolution_sca(Di, steps, uniname, saveFigures, xaxis_type = 'T', scasize = 1)
                #draw_tilt_evolution_sca(T, steps, uniname, saveFigures, xaxis_type = 'T', scasize = 1)
                self.Dobj = draw_dist_evolution(Di, steps, Tgrad = self.Tgrad, uniname=uniname, saveFigures = saveFigures, xaxis_type = 'T', Ti = Ti,invert_x=invert_x) 
                self.Tobj = draw_tilt_evolution(T, steps, Tgrad = self.Tgrad, uniname=uniname, saveFigures = saveFigures, xaxis_type = 'T', Ti = Ti,invert_x=invert_x) 
                
        else: # read_mode 1, constant-T MD (equilibration)
            from pdyna.analysis import draw_dist_density, draw_tilt_density, draw_conntype_tilt_density
            Dmu,Dstd = draw_dist_density(Di, uniname, saveFigures, n_bins = 100, title=None)
            if not tilt_corr_NN1:
                if structure_type == 1 or not hasattr(self, 'octahedral_connectivity'):
                    draw_tilt_density(T, uniname, saveFigures,symm_n_fold=symm_n_fold,title=title)
                elif structure_type in (2,3):
                    oc = self.octahedral_connectivity
                    if len(oc) == 1:
                        title = list(oc.keys())[0]+", "+title
                        draw_tilt_density(T, uniname, saveFigures,symm_n_fold=symm_n_fold,title=title)
                    else:
                        title = "mixed, "+title
                        draw_conntype_tilt_density(T, oc, uniname, saveFigures,symm_n_fold=symm_n_fold,title=title)
            
            self.prop_lib['distortion'] = [Dmu,Dstd]
            Tval = np.array(compute_tilt_density(T,plot_fitting=False)).reshape((3,-1))
            self.prop_lib['tilting'] = Tval
            
            
        # autocorr
        if tiltautoCorr:
            from pdyna.analysis import fit_exp_decay_both, fit_exp_decay_both_correct, Tilt_correlation
            xt, yt = Tilt_correlation(T,self.Timestep,False)
            kt = []
            try:
                for i in range(3):
                    kt.append(fit_exp_decay_both(xt, yt[:,i]))
            except RuntimeError:
                kt = []
                for i in range(3):
                    kt.append(fit_exp_decay_both_correct(xt, yt[:,i]))

            kt = np.array(kt)
            self.tilt_autocorr = kt
            self.prop_lib['tilt_autocorr'] = kt
        
        # NN1 correlation function of tilting (Glazer notation)
        if tilt_corr_NN1:
            from pdyna.analysis import abs_sqrt, draw_tilt_corr_evolution_sca, draw_tilt_and_corr_density_shade
            
            Benv = self._Benv
                
            if Benv.shape[1] == 3: # indicate a 2*2*2 supercell
                
                #ref_coords = np.array([[0,0,0],[-6.15,0,0],[0,-6.15,0],[0,0,-6.15]])
                if not self._non_orthogonal:
                    for i in range(Benv.shape[0]):
                        # for each Pb atom find its nearest Pb in each orthogonal direction. 
                        orders = np.argmax(np.abs(Bpos[0,Benv[i,:],:] - Bpos[0,i,:]), axis=0)
                        Benv[i,:] = Benv[i,:][orders] # correct the order of coords by 'x,y,z'
                    
                    # now each row of Benv contains the Pb atom index that sit in x,y and z direction of the row-numbered Pb atom.
                    Corr = np.empty((T.shape[0],T.shape[1],3))
                    for B1 in range(T.shape[1]):
                            
                        Corr[:,B1,0] = abs_sqrt(T[:,B1,0]*T[:,Benv[B1,0],0])
                        Corr[:,B1,1] = abs_sqrt(T[:,B1,1]*T[:,Benv[B1,1],1])
                        Corr[:,B1,2] = abs_sqrt(T[:,B1,2]*T[:,Benv[B1,2],2])
                    
                else:
                    mm = np.linalg.inv(self._rotmat_from_orthogonal)
                    for i in range(Benv.shape[0]):
                        tempv = np.matmul(Bpos[0,Benv[i,:],:] - Bpos[0,i,:],mm)
                        orders = np.argmax(np.abs(tempv), axis=0)
                        Benv[i,:] = Benv[i,:][orders] # correct the order of coords by 'x,y,z'
                    
                    Corr = np.empty((T.shape[0],T.shape[1],3))
                    for B1 in range(T.shape[1]):
                            
                        Corr[:,B1,0] = abs_sqrt(T[:,B1,0]*T[:,Benv[B1,0],0])
                        Corr[:,B1,1] = abs_sqrt(T[:,B1,1]*T[:,Benv[B1,1],1])
                        Corr[:,B1,2] = abs_sqrt(T[:,B1,2]*T[:,Benv[B1,2],2])
                
                # Normalize Corr
                #Corr = Corr/np.amax(Corr)

            elif Benv.shape[1] == 6: # indicate a larger supercell
                
                Bcoordenv = self._Bcoordenv
                                
                ref_octa = np.array([[1,0,0],[-1,0,0],
                                     [0,1,0],[0,-1,0],
                                     [0,0,1],[0,0,-1]])
                if self._non_orthogonal:
                    ref_octa = np.matmul(ref_octa,self._rotmat_from_orthogonal)
                for i in range(Bcoordenv.shape[0]):
                    orders = np.zeros((1,6))
                    for j in range(6):
                        orders[0,j] = np.argmax(np.dot(Bcoordenv[i,:,:],ref_octa[j,:]))
                    Benv[i,:] = Benv[i,:][orders.astype(int)]
                        
                # now each row of Benv contains the Pb atom index that sit in x,y and z direction of the row-numbered Pb atom.
                Corr = np.empty((T.shape[0],T.shape[1],6))
                for B1 in range(T.shape[1]):
                        
                    Corr[:,B1,[0,1]] = abs_sqrt(T[:,[B1],0]*T[:,Benv[B1,[0,1]],0]) # x neighbour 1,2
                    Corr[:,B1,[2,3]] = abs_sqrt(T[:,[B1],1]*T[:,Benv[B1,[2,3]],1]) # y neighbour 1,2
                    Corr[:,B1,[4,5]] = abs_sqrt(T[:,[B1],2]*T[:,Benv[B1,[4,5]],2]) # z neighbour 1,2
                
            else: 
                raise TypeError(f"The environment matrix is incorrect. {Benv.shape[1]} ")
            
            self._BNNenv = Benv
            self.Tilting_Corr = Corr
            
            if Benv.shape[1] == 6 and full_NN1_corr:
                from pdyna.analysis import draw_tilt_and_corr_density_full,draw_tilt_coaxial
                Cf = np.empty((T.shape[0],T.shape[1],3,3,2))
                for B1 in range(T.shape[1]):
                        
                    Cf[:,B1,0,0,:] = abs_sqrt(T[:,[B1],0]*T[:,Benv[B1,[0,1]],0]) # x neighbour 1,2
                    Cf[:,B1,1,1,:] = abs_sqrt(T[:,[B1],1]*T[:,Benv[B1,[2,3]],1]) # y neighbour 1,2
                    Cf[:,B1,2,2,:] = abs_sqrt(T[:,[B1],2]*T[:,Benv[B1,[4,5]],2]) # z neighbour 1,2
                    Cf[:,B1,0,1,:] = abs_sqrt(T[:,[B1],0]*T[:,Benv[B1,[2,3]],0]) # x neighbour 1,2
                    Cf[:,B1,0,2,:] = abs_sqrt(T[:,[B1],0]*T[:,Benv[B1,[4,5]],0]) # y neighbour 1,2
                    Cf[:,B1,1,0,:] = abs_sqrt(T[:,[B1],1]*T[:,Benv[B1,[0,1]],1]) # z neighbour 1,2
                    Cf[:,B1,1,2,:] = abs_sqrt(T[:,[B1],1]*T[:,Benv[B1,[4,5]],1]) # x neighbour 1,2
                    Cf[:,B1,2,0,:] = abs_sqrt(T[:,[B1],2]*T[:,Benv[B1,[0,1]],2]) # y neighbour 1,2
                    Cf[:,B1,2,1,:] = abs_sqrt(T[:,[B1],2]*T[:,Benv[B1,[2,3]],2]) # z neighbour 1,2
                    
                draw_tilt_and_corr_density_full(T,Cf,uniname,saveFigures,title=title)
                draw_tilt_coaxial(T,uniname,saveFigures,title=title)
                
            if read_mode == 2 and self.Tgrad == 0:
                self.Cobj = draw_tilt_corr_density_time(T, self.Tilting_Corr, timeline, uniname, saveFigures=False, smoother=smoother)
            
            elif self.Tgrad != 0:
                pass
                #draw_tilt_corr_evolution_sca(Corr, steps, uniname, saveFigures, xaxis_type = 'T') 
                
            else:
                polarity = draw_tilt_and_corr_density_shade(T,Corr, uniname, saveFigures,title=title)
                self.prop_lib["tilt_corr_polarity"] = polarity
                
                # justify if there is a true split of tilting peaks with TCP
                Tval = np.array(compute_tilt_density(T,plot_fitting=True,corr_vals=polarity)).reshape((3,-1))
                self.prop_lib['tilting'] = Tval
        
        if tilt_domain:
            from pdyna.analysis import compute_tilt_domain
            compute_tilt_domain(Corr, self.Timestep, uniname, saveFigures)

        if tilt_corr_spatial:
            import math
            from scipy.stats import binned_statistic_dd as binstat
            from pdyna.analysis import draw_tilt_spatial_corr, quantify_tilt_domain

            cc = self.st0.frac_coords[self.Bindex,:]
            
            supercell_size = self.supercell_size
            
            boost = list(np.where(np.logical_or(np.abs(np.amin(cc,axis=0))*supercell_size<0.1,np.abs(np.amax(cc,axis=0))*supercell_size>supercell_size-0.1))[0])
            if len(boost) > 0:
                for b in boost:
                    cc[:,b] = cc[:,b]+1/supercell_size/2
            cc[cc>1] = cc[cc>1]-1
            
            clims = np.array([[(np.quantile(cc[:,0],1/(supercell_size**2))+np.amin(cc[:,0]))/2,(np.quantile(cc[:,0],1-1/(supercell_size**2))+np.amax(cc[:,0]))/2],
                              [(np.quantile(cc[:,1],1/(supercell_size**2))+np.amin(cc[:,1]))/2,(np.quantile(cc[:,1],1-1/(supercell_size**2))+np.amax(cc[:,1]))/2],
                              [(np.quantile(cc[:,2],1/(supercell_size**2))+np.amin(cc[:,2]))/2,(np.quantile(cc[:,2],1-1/(supercell_size**2))+np.amax(cc[:,2]))/2]])
            
            bin_indices = binstat(cc, None, 'count', bins=[supercell_size,supercell_size,supercell_size], 
                                  range=[[clims[0,0]-0.5*(1/supercell_size), 
                                          clims[0,1]+0.5*(1/supercell_size)], 
                                         [clims[1,0]-0.5*(1/supercell_size), 
                                          clims[1,1]+0.5*(1/supercell_size)],
                                         [clims[2,0]-0.5*(1/supercell_size), 
                                          clims[2,1]+0.5*(1/supercell_size)]],
                                  expand_binnumbers=True).binnumber
            # validate the binning
            atom_indices = np.array([bin_indices[0,i]+(bin_indices[1,i]-1)*supercell_size+(bin_indices[2,i]-1)*supercell_size**2 for i in range(bin_indices.shape[1])])
            bincount = np.unique(atom_indices, return_counts=True)[1]
            if len(bincount) != supercell_size**3:
                raise TypeError("Incorrect number of bins. ")
            if max(bincount) != min(bincount):
                raise ValueError("Not all bins contain exactly the same number of atoms (1). ")
            
# =============================================================================
#             B3denv = np.empty((3,supercell_size**2,supercell_size))
#             for i in range(3):
#                 dims = [0,1,2]
#                 dims.remove(i)
#                 binx = bin_indices[dims,:]
#                 
#                 binned = [[] for _ in range(supercell_size**2)]
#                 for j in range(binx.shape[1]):
#                     binned[(binx[0,j]-1)+supercell_size*(binx[1,j]-1)].append(j)
#                 binned = np.array(binned)
# 
#                 
#                 for j in range(binned.shape[0]): # sort each bin in "i" direction coords
#                     binned[j,:] = np.array([x for _, x in sorted(zip(cc[binned[j,:],i], binned[j,:]))]).reshape(1,supercell_size)
#                 
#                 B3denv[i,:] = binned
#             
#             B3denv = B3denv.astype(int)
#             
#             num_nn = math.ceil((supercell_size-1)/2)
# 
# 
#             spatialnn = []
#             spatialnorm = []
#             for space in range(3):
#                 layernn = []
#                 layernorm = []
#                 for layer in range(supercell_size):
#                     corrnn = np.empty((num_nn+1,3))
#                     corrnorm = np.empty((num_nn+1,3))
#                     for nn in range(num_nn+1):
#                         #pos2 = layer+(nn)
#                         #if pos2 > supercell_size-1:
#                         #    pos2 -= (supercell_size)
#                         pos1 = layer-(nn)
#                         
#                         tc1 = np.multiply(T[:,B3denv[space,:,layer],:],T[:,B3denv[space,:,pos1],:]).reshape(-1,3)
#                         #tc2 = np.multiply(T[:,B3denv[space,:,layer],:],T[:,B3denv[space,:,pos2],:]).reshape(-1,3)
#                         tc = np.nanmean(tc1,axis=0)[np.newaxis,:]
#                         #tc = np.nanmean(np.concatenate((tc1,tc2),axis=0),axis=0)[np.newaxis,:]
#                         tcnorm = (np.sqrt(np.abs(tc))*np.sign(tc))
#                         corrnn[nn,:] = tc
#                         corrnorm[nn,:] = tcnorm
#                     layernn.append(corrnn)
#                     layernorm.append(corrnorm)
#                 layernn = np.nanmean(np.array(layernn),axis=0)
#                 layernn = layernn/layernn[0,:]
#                 spatialnn.append(layernn)
#                 layernorm = np.nanmean(np.array(layernorm),axis=0)
#                 layernorm = layernorm/layernorm[0,:]
#                 spatialnorm.append(layernorm)
#                 
#             spatialnn = np.array(spatialnn)
#             spatialnorm = np.array(spatialnorm)
# =============================================================================

            #thr = 3
            bin_indices = bin_indices-1 # 0-indexing
            
            num_nn = math.ceil((supercell_size-1)/2)
            scmnorm = np.empty((bin_indices.shape[1],3,num_nn+1,3))
            scmnorm[:] = np.nan
            scm = np.empty((bin_indices.shape[1],3,num_nn+1,3))
            scm[:] = np.nan
            for o in range(bin_indices.shape[1]):
                si = bin_indices[:,[o]]
                for space in range(3):
                    for n in range(num_nn+1):
                        addit = np.array([[0],[0],[0]])
                        addit[space,0] = n
                        pos1 = si + addit
                        pos1[pos1>supercell_size-1] = pos1[pos1>supercell_size-1]-supercell_size
                        k1 = np.where(np.all(bin_indices==pos1,axis=0))[0][0]
                        tc1 = np.multiply(T[:,o,:],T[:,k1,:])
                        #if thr != 0:
                        #    tc1[np.abs(T[:,o,:])<thr] = np.nan
                        #tc1norm = np.sqrt(np.abs(tc1))*np.sign(tc1)
                        tc = np.nanmean(tc1,axis=0)
                        #tcnorm = np.nanmean(tc1norm,axis=0)
                        tcnorm = np.sqrt(np.abs(tc))*np.sign(tc)
                        scmnorm[o,space,n,:] = tcnorm
                        scm[o,space,n,:] = tc
                    
                        
            scm = scm/scm[:,:,[0],:]
            scmnorm = scmnorm/scmnorm[:,:,[0],:]
            spatialnn = np.mean(scm,axis=0)
            spatialnorm = np.mean(scmnorm,axis=0)
            self.spatialCorr = {'raw':scm,'norm':scmnorm}           
            
            scdecay = quantify_tilt_domain(spatialnn,spatialnorm) # spatial decay length in 'unit cell'
            self.spatialCorrLength = scdecay  
            self.prop_lib["spatial_corr_length"] = scdecay 
            print(f"Tilting spatial correlation length: \n {np.round(scdecay,3)}")
            
# =============================================================================
#             # correlation in the 110 directions
#             scmnorm = np.empty((bin_indices.shape[1],3,num_nn+1,3))
#             scmnorm[:] = np.nan
#             scm = np.empty((bin_indices.shape[1],3,num_nn+1,3))
#             scm[:] = np.nan
#             for o in range(bin_indices.shape[1]):
#                 si = bin_indices[:,[o]]
#                 for space in range(3):
#                     for n in range(num_nn+1):
#                         addit = np.array([[n],[n],[n]])
#                         addit[space,0] = 0
#                         a1 = addit.copy()
#                         addit[space-1,0] = addit[space-1,0]*(-1)
#                         a2 = addit.copy()
#                         
#                         pos1 = si + a1
#                         pos1[pos1>supercell_size-1] = pos1[pos1>supercell_size-1]-supercell_size
#                         pos1[pos1<0] = pos1[pos1<0]+supercell_size
#                         k1 = np.where(np.all(bin_indices==pos1,axis=0))[0][0]
#                         tc1 = np.multiply(T[:,o,:],T[:,k1,:])
#                         
#                         pos2 = si + a2
#                         pos2[pos2>supercell_size-1] = pos2[pos2>supercell_size-1]-supercell_size
#                         pos2[pos2<0] = pos2[pos2<0]+supercell_size
#                         k2 = np.where(np.all(bin_indices==pos2,axis=0))[0][0]
#                         tc2 = np.multiply(T[:,o,:],T[:,k2,:])
#                         
#                         #tc1norm = np.sqrt(np.abs(tc1))*np.sign(tc1)
#                         tc = np.nanmean(np.concatenate((tc1,tc2),axis=0),axis=0)
#                         #tcnorm = np.nanmean(tc1norm,axis=0)
#                         tcnorm = np.sqrt(np.abs(tc))*np.sign(tc)
#                         scmnorm[o,space,n,:] = tcnorm
#                         scm[o,space,n,:] = tc
#                         
#             scm = scm/scm[:,:,[0],:]
#             scmnorm = scmnorm/scmnorm[:,:,[0],:]
#             spatialnn = np.mean(scm,axis=0)
#             spatialnorm = np.mean(scmnorm,axis=0)
#             #self.spatialCorr110 = {'raw':scm,'norm':scmnorm}           
#             
#             scdecay110 = quantify_tilt_domain(spatialnn,spatialnorm)*np.sqrt(2) # spatial decay length in 'unit cell'
#             self.spatialCorrLength110 = scdecay110  
#             self.prop_lib["spatial_corr_length_110"] = scdecay110 
#             print(f"Tilting spatial correlation length in (110) directions: \n {np.round(scdecay110,3)}")
# =============================================================================
            
        
        if vis3D_domain != 0 and vis3D_domain in (1,2):
            import matplotlib.pyplot as plt
            from scipy.stats import binned_statistic_dd as binstat
            from pdyna.analysis import savitzky_golay, vis3D_domain_anime

            axisvis = 2

            cc = self.st0.frac_coords[self.Bindex,:]
            
            supercell_size = self.supercell_size
            
            clims = np.array([[(np.quantile(cc[:,0],1/(supercell_size**2))+np.amin(cc[:,0]))/2,(np.quantile(cc[:,0],1-1/(supercell_size**2))+np.amax(cc[:,0]))/2],
                              [(np.quantile(cc[:,1],1/(supercell_size**2))+np.amin(cc[:,1]))/2,(np.quantile(cc[:,1],1-1/(supercell_size**2))+np.amax(cc[:,1]))/2],
                              [(np.quantile(cc[:,2],1/(supercell_size**2))+np.amin(cc[:,2]))/2,(np.quantile(cc[:,2],1-1/(supercell_size**2))+np.amax(cc[:,2]))/2]])
            
            bin_indices = binstat(cc, None, 'count', bins=[supercell_size,supercell_size,supercell_size], 
                                  range=[[clims[0,0]-0.5*(1/supercell_size), 
                                          clims[0,1]+0.5*(1/supercell_size)], 
                                         [clims[1,0]-0.5*(1/supercell_size), 
                                          clims[1,1]+0.5*(1/supercell_size)],
                                         [clims[2,0]-0.5*(1/supercell_size), 
                                          clims[2,1]+0.5*(1/supercell_size)]],
                                  expand_binnumbers=True).binnumber
            # validate the binning
            atom_indices = np.array([bin_indices[0,i]+(bin_indices[1,i]-1)*supercell_size+(bin_indices[2,i]-1)*supercell_size**2 for i in range(bin_indices.shape[1])])
            bincount = np.unique(atom_indices, return_counts=True)[1]
            if len(bincount) != supercell_size**3:
                raise TypeError("Incorrect number of bins. ")
            if max(bincount) != min(bincount):
                raise ValueError("Not all bins contain exactly the same number of atoms (1). ")
            
            # visualize tilt domains
            if vis3D_domain == 2:

                temp1 = Corr[:,:,[axisvis*2]]
                temp1[np.abs(temp1)>25] = 0
                temp1[np.abs(temp1)>15] = np.sign(temp1)[np.abs(temp1)>15]*15
                temp2 = Corr[:,:,[axisvis*2+1]]
                temp2[np.abs(temp2)>25] = 0
                temp2[np.abs(temp2)>15] = np.sign(temp2)[np.abs(temp2)>15]*15
                
                time_window = 5 # picosecond
                sgw = round(time_window/self.Timestep)
                if sgw<5: sgw = 5
                if sgw%2==0: sgw+=1
                
                for i in range(temp1.shape[1]):
                    temp = temp1[:,i,0]
                    temp = savitzky_golay(temp,window_size=sgw)
                    temp1[:,i,0] = temp
                    temp = temp2[:,i,0]
                    temp = savitzky_golay(temp,window_size=sgw)
                    temp2[:,i,0] = temp
                
                polfeat = (np.power(np.abs(temp1),2)*np.sign(temp1)+np.power(np.abs(temp2),2)*np.sign(temp2))/2
                polfeat = np.power(np.abs(polfeat),1/2)*np.sign(polfeat)
                
                polfeat = polfeat/np.amax(np.abs(polfeat))
                polfeat = np.clip(polfeat,-0.8,0.8)
                polfeat = polfeat/np.amax(np.abs(polfeat))
                plotfeat = polfeat.copy()

                plotfeat = np.power(np.abs(plotfeat),1/2)*np.sign(plotfeat)
                plotfeat = np.clip(plotfeat,-0.8,0.8)
                plotfeat = plotfeat/np.amax(np.abs(plotfeat))
                
                figname1 = f"Tiltcorr3D_domain_{uniname}"
                
            
            elif vis3D_domain == 1:
                
                ampfeat = T.copy()
                ampfeat[np.isnan(ampfeat)] = 0
                ampfeat[np.logical_or(ampfeat>23,ampfeat<-23)] = 0
                plotfeat = ampfeat[:,:,[axisvis]]
                
                time_window = 2 # picosecond
                sgw = round(time_window/self.Timestep)
                if sgw<5: sgw = 5
                if sgw%2==0: sgw+=1
                
                for i in range(plotfeat.shape[1]):
                    temp = plotfeat[:,i,0]
                    temp = savitzky_golay(temp,window_size=sgw)
                    plotfeat[:,i,0] = temp
                
                #plotfeat = np.sqrt(np.abs(plotfeat))*np.sign(plotfeat)
                clipedges = (np.quantile(plotfeat,0.92)-np.quantile(plotfeat,0.08))/2
                clipedges1 = [-2,2]
                plotfeat = np.clip(plotfeat,-clipedges,clipedges)
                plotfeat[np.logical_and(plotfeat<clipedges1[1],plotfeat>clipedges1[0])] = 0
                
                figname1 = f"Tilt3D_domain_{uniname}"
            

            def map_rgb_tilt(x):
                x = x/np.amax(np.abs(x))/2+0.5
                return plt.cm.coolwarm(x)[:,:,0,:]
            
            cfeat = map_rgb_tilt(plotfeat)
            
            readtime = 60 # ps
            readfreq = 0.2 # ps
            fstart = round(cfeat.shape[0]*0.1)
            fend = min(fstart+round(readtime/self.Timestep),round(cfeat.shape[0]*0.9))
            frs = list(np.round(np.linspace(fstart,fend,round((fend-fstart)*self.Timestep/readfreq)+1)).astype(int))
            
            et0 = time.time()
            vis3D_domain_anime(cfeat,frs,self.Timestep,supercell_size,bin_indices,figname1)
            et1 = time.time()
            print("Tilt domain visualization took:",round((et1-et0)/60,2),"minutes. ")
        
            
# =============================================================================
#             # visualize distortion domains
#             dlims = [np.nanquantile(Di[:,:,i],0.9) for i in range(4)]
#             dmeans = [np.nanmean(Di[:,:,i]) for i in range(4)]
#             Df = Di.copy()
#             for i in range(4):
#                 #mask = np.logical_or(Df[:,:,i]>dlims[i],np.isnan(Df[:,:,i]))
#                 mask1 = np.isnan(Df[:,:,i])
#                 mask2 = Df[:,:,i]>dlims[i]
#                 Df[:,:,i][mask1] = dmeans[i]
#                 Df[:,:,i][mask2] = dlims[i]
#             plotfeat = np.linalg.norm(Df,axis=2)
#             #plotfeat = Df[:,:,3]
#             
#             time_window = 5 # picosecond
#             sgw = round(time_window/self.Timestep)
#             if sgw<5: sgw = 5
#             if sgw%2==0: sgw+=1
#             
#             for i in range(plotfeat.shape[1]):
#                 temp = plotfeat[:,i]
#                 temp = savitzky_golay(temp,window_size=sgw)
#                 plotfeat[:,i] = temp
#             
#             plotfeat[plotfeat<np.quantile(plotfeat,0.05)] = np.quantile(plotfeat,0.05)
#             plotfeat[plotfeat>np.quantile(plotfeat,0.95)] = np.quantile(plotfeat,0.95)
#             plotfeat = plotfeat-np.amin(plotfeat)
#             plotfeat = plotfeat/np.amax(plotfeat)
#             plotfeat = np.power(np.abs(plotfeat-0.5),2/3)*np.sign(plotfeat-0.5)
#             
#             figname1 = f"Dist3_3D_domain_{uniname}"
# 
#             def map_rgb_dist(x):
#                 x = x/np.amax(np.abs(x))/2+0.5
#                 return plt.cm.YlGnBu(x)
#             
#             cfeat = map_rgb_dist(plotfeat)
#             
#             readtime = 60 # ps
#             readfreq = 0.6 # ps
#             fstart = round(cfeat.shape[0]*0.1)
#             fend = min(fstart+round(readtime/self.Timestep),round(cfeat.shape[0]*0.9))
#             frs = list(np.round(np.linspace(fstart,fend,round((fend-fstart)*self.Timestep/readfreq)+1)).astype(int))
#             
#             et0 = time.time()
#             vis3D_domain_anime(cfeat,frs,self.Timestep,supercell_size,bin_indices,figname1)
#             et1 = time.time()
#             print("Distortion domain visualization took:",round((et1-et0)/60,2),"minutes. ")
# =============================================================================
        

        et1 = time.time()
        self.timing["tilt_distort"] = et1-et0
        
    
    def molecular_orientation(self,uniname,read_mode,allow_equil,MOautoCorr,MO_corr_spatial,title,saveFigures,smoother,draw_MO_anime):
        """
        A-site molecular orientation (MO) analysis.

        Parameters
        ----------
        MOautoCorr : calculate MO decorrelation time constant
        MO_corr_spatial : compute spatial MO correlation functions

        """
        
        et0 = time.time()
        
        from pdyna.analysis import MO_correlation, orientation_density, orientation_density_2pan, fit_exp_decay, orientation_density_3D_sphere
        from pdyna.structural import apply_pbc_cart_vecs
        
        Afa = self.A_sites["FA"]
        Ama = self.A_sites["MA"]
        Aazr = self.A_sites["Azr"]
        
        Cpos = self.Allpos[:,self.Cindex,:]
        Npos = self.Allpos[:,self.Nindex,:]
        
        trajnum = list(range(round(self.nframe*allow_equil),self.nframe))
        latmat = self.latmat[trajnum,:]
        r0=distance_matrix_handler(Cpos[0,:],Npos[0,:],self.latmat[0,:],self.at0.cell,self.at0.pbc,self.complex_pbc)
        r1=distance_matrix_handler(Cpos[-1,:],Npos[-1,:],self.latmat[-1,:],self.at0.cell,self.at0.pbc,self.complex_pbc)
        CNdiff = np.amax(np.abs(r0-r1))
        
        if CNdiff > 5:
            print(f"!MO: A change of C-N connectivity is detected (ref value {np.round(CNdiff,3)} A).")

        MOvecs = {}
        MOcenter = {}
        
        if len(Ama) > 0:
            Clist = [i[0][0] for i in Ama]
            Nlist = [i[1][0] for i in Ama]
            
            Cpos = self.Allpos[trajnum,:][:,Clist,:]
            Npos = self.Allpos[trajnum,:][:,Nlist,:]
            
            cn = Cpos-Npos
            cn = apply_pbc_cart_vecs(cn,latmat)
            CN = np.divide(cn,np.expand_dims(np.linalg.norm(cn,axis=2),axis=2))
            
            MOvecs["MA"] = cn
            MA_MOvecnorm = CN
            MA_center = Cpos - 7/13*cn
            MOcenter["MA"] = MA_center
            
            orientation_density(CN,"MA",saveFigures,uniname,title=title)
            if draw_MO_anime and saveFigures:
                orientation_density_3D_sphere(CN,"MA",saveFigures,uniname)
            
            
        if len(Afa) > 0:
            
            Clist = [i[0][0] for i in Afa]
            N1list = [i[1][0] for i in Afa]
            N2list = [i[1][1] for i in Afa]
            
            Nlist = N1list+N2list
            
            Cpos = self.Allpos[trajnum,:][:,Clist,:]
            N1pos = self.Allpos[trajnum,:][:,N1list,:]
            N2pos = self.Allpos[trajnum,:][:,N2list,:]
            
            cn1 = Cpos-N1pos
            cn2 = Cpos-N2pos
            cn1 = apply_pbc_cart_vecs(cn1,latmat)
            cn2 = apply_pbc_cart_vecs(cn2,latmat)
            
            cn = (cn1+cn2)/2
            CN = np.divide(cn,np.expand_dims(np.linalg.norm(cn,axis=2),axis=2))
            
            nn = N1pos-N2pos
            nn = apply_pbc_cart_vecs(nn,latmat)
            NN = np.divide(nn,np.expand_dims(np.linalg.norm(nn,axis=2),axis=2))
            
            MOvecs["FA1"] = cn
            MOvecs["FA2"] = nn
            FA_MOvec1norm = CN
            FA_MOvec2norm = NN
            FA_center = Cpos - 7/10*cn
            MOcenter["FA"] = FA_center
            
            orientation_density_2pan(CN,NN,"FA",saveFigures,uniname,title=title)
            if draw_MO_anime and saveFigures:
                orientation_density_3D_sphere(CN,"FA1",saveFigures,uniname)
                orientation_density_3D_sphere(NN,"FA2",saveFigures,uniname)
        
            
        if len(Aazr) > 0:
            
            C1list = [i[0][0] for i in Aazr]
            C2list = [i[0][1] for i in Aazr]
            Nlist = [i[1][0] for i in Aazr]
            
            Clist = C1list+C2list
            
            C1pos = self.Allpos[trajnum,:][:,C1list,:]
            C2pos = self.Allpos[trajnum,:][:,C2list,:]
            Npos = self.Allpos[trajnum,:][:,Nlist,:]
            
            cn1 = C1pos-Npos
            cn2 = C2pos-Npos
            cn1 = apply_pbc_cart_vecs(cn1,latmat)
            cn2 = apply_pbc_cart_vecs(cn2,latmat)
            
            cn = (cn1+cn2)/2
            CN = np.divide(cn,np.expand_dims(np.linalg.norm(cn,axis=2),axis=2))
            
            nn = C1pos-C2pos
            nn = apply_pbc_cart_vecs(nn,latmat)
            NN = np.divide(nn,np.expand_dims(np.linalg.norm(nn,axis=2),axis=2))
            
            MOvecs["Azr1"] = cn
            MOvecs["Azr2"] = nn
            Azr_MOvec1norm = CN
            Azr_MOvec2norm = NN
            Azr_center = Npos + 12/19*cn
            MOcenter["Azr"] = Azr_center
            
            orientation_density_2pan(CN,NN,"Azr",saveFigures,uniname,title=title)
            if draw_MO_anime and saveFigures:
                orientation_density_3D_sphere(CN,"Azr1",saveFigures,uniname)
                orientation_density_3D_sphere(NN,"Azr2",saveFigures,uniname)
        
        # save the MO vectors together
        self.MOvector = MOvecs
        self.MOcenter = MOcenter
        
        # total polarization
        if not (len(Afa) > 0 and len(Ama) > 0 and len(Aazr) > 0):
            if len(Ama) > 0: v=self.MOvector['MA']
            if len(Afa) > 0: v=self.MOvector['FA1']
            if len(Aazr) > 0: v=self.MOvector['Azr1']
            
            v=np.divide(v,np.expand_dims(np.linalg.norm(v,axis=2),axis=2))
            v=v.reshape(-1,3)
            b=np.sum(v,axis=0)/v.shape[0]
            self.total_MO_polarization = b

        if MOautoCorr is True:
            tconst = {}
            if sum([len(Afa)>0,len(Ama)>0,len(Aazr)>0])>1 :
                raise TypeError("Need to write code for both species here")
            #sys.stdout.flush()
            
            if len(Ama) > 0:
                corrtime, autocorr = MO_correlation(MA_MOvecnorm,self.MDTimestep,False,uniname)
                self.MO_MA_autocorr = np.concatenate((corrtime,autocorr),axis=0)
                tconst_MA = fit_exp_decay(corrtime, autocorr)
                print("MO: MA decorrelation time: "+str(round(tconst_MA,4))+' ps')
                if tconst_MA < 0:
                    print("!MO: Negative decorrelation MA time constant is found, please check if the trajectory is too short or system size too small. ")
                tconst["MA"] = tconst_MA
                
            if len(Afa) > 0:
                corrtime, autocorr = MO_correlation(FA_MOvec1norm,self.MDTimestep,False,uniname)
                self.MO_FA1_autocorr = np.concatenate((corrtime,autocorr),axis=0)
                tconst_FA1 = fit_exp_decay(corrtime, autocorr)
                print("MO: FA1 decorrelation time: "+str(round(tconst_FA1,4))+' ps')
                if tconst_FA1 < 0:
                    print("!MO: Negative decorrelation FA1 time constant is found, please check if the trajectory is too short or system size too small. ")
                    
                corrtime, autocorr = MO_correlation(FA_MOvec2norm,self.MDTimestep,False,uniname)
                self.MO_FA2_autocorr = np.concatenate((corrtime,autocorr),axis=0)
                tconst_FA2 = fit_exp_decay(corrtime, autocorr)
                print("MO: FA2 decorrelation time: "+str(round(tconst_FA2,4))+' ps')
                if tconst_FA2 < 0:
                    print("!MO: Negative decorrelation FA2 time constant is found, please check if the trajectory is too short or system size too small. ")
                    
                tconst["FA"] = [tconst_FA1,tconst_FA2]
            
            if len(Aazr) > 0:
                corrtime, autocorr = MO_correlation(Azr_MOvec1norm,self.MDTimestep,False,uniname)
                self.MO_Azr1_autocorr = np.concatenate((corrtime,autocorr),axis=0)
                tconst_Azr1 = fit_exp_decay(corrtime, autocorr)
                print("MO: Azr1 decorrelation time: "+str(round(tconst_Azr1,4))+' ps')
                if tconst_Azr1 < 0:
                    print("!MO: Negative decorrelation Azr1 time constant is found, please check if the trajectory is too short or system size too small. ")
                    
                corrtime, autocorr = MO_correlation(Azr_MOvec2norm,self.MDTimestep,False,uniname)
                self.MO_Azr2_autocorr = np.concatenate((corrtime,autocorr),axis=0)
                tconst_Azr2 = fit_exp_decay(corrtime, autocorr)
                print("MO: Azr2 decorrelation time: "+str(round(tconst_Azr2,4))+' ps')
                if tconst_Azr2 < 0:
                    print("!MO: Negative decorrelation Azr2 time constant is found, please check if the trajectory is too short or system size too small. ")
                    
                tconst["Azr"] = [tconst_Azr1,tconst_Azr2]
            
            self.MOlifetime = tconst
            self.prop_lib['reorientation'] = tconst
            print(" ")
       
        
        if MO_corr_spatial and not (len(Afa) > 0 and len(Ama) > 0 and len(Aazr) > 0):
            import math
            from pdyna.structural import get_frac_from_cart
            from pdyna.analysis import draw_MO_order_time, draw_MO_spatial_corr_NN12, draw_MO_spatial_corr, draw_MO_spatial_corr_norm_var
            from scipy.stats import binned_statistic_dd as binstat
            
            supercell_size = self.supercell_size
            if len(Ama) > 0: MOcent = MA_center
            if len(Afa) > 0: MOcent = FA_center
            if len(Aazr) > 0: MOcent = Azr_center
            
            safety_margin = np.array([1-1/supercell_size,1])
            
            cc = get_frac_from_cart(MOcent, latmat)[0,:]
            
            rect = [0,0,0]
            if np.argmin(np.abs(safety_margin - (np.amax(cc[:,0])-np.amin(cc[:,0])))): 
                cc[:,0] = cc[:,0]-(np.amax(cc[:,0])-(1-1/supercell_size/2))
                rect[0] = 1
            if np.argmin(np.abs(safety_margin - (np.amax(cc[:,1])-np.amin(cc[:,1])))): 
                cc[:,1] = cc[:,1]-(np.amax(cc[:,1])-(1-1/supercell_size/2))
                rect[1] = 1
            if np.argmin(np.abs(safety_margin - (np.amax(cc[:,2])-np.amin(cc[:,2])))): 
                cc[:,2] = cc[:,2]-(np.amax(cc[:,2])-(1-1/supercell_size/2))
                rect[2] = 1
            
            for i in range(3):
                if rect[i] == 0:
                    if np.amin(cc[:,i]) < 1/supercell_size/4:
                        cc[:,i] = cc[:,i] + 1/supercell_size/8*3
                    if np.amin(cc[:,i]) > 1-1/supercell_size/4:
                        cc[:,i] = cc[:,i] - 1/supercell_size/8*3
            
            for i in range(cc.shape[0]):
                for j in range(cc.shape[1]):
                    if cc[i,j] > 1:
                        cc[i,j] = cc[i,j]-1
                    if cc[i,j] < 0:
                        cc[i,j] = cc[i,j]+1
            
            clims = np.array([[(np.quantile(cc[:,0],1/(supercell_size**2))+np.amin(cc[:,0]))/2,(np.quantile(cc[:,0],1-1/(supercell_size**2))+np.amax(cc[:,0]))/2],
                              [(np.quantile(cc[:,1],1/(supercell_size**2))+np.amin(cc[:,1]))/2,(np.quantile(cc[:,1],1-1/(supercell_size**2))+np.amax(cc[:,1]))/2],
                              [(np.quantile(cc[:,2],1/(supercell_size**2))+np.amin(cc[:,2]))/2,(np.quantile(cc[:,2],1-1/(supercell_size**2))+np.amax(cc[:,2]))/2]])
            
            bin_indices = binstat(cc, None, 'count', bins=[supercell_size,supercell_size,supercell_size], 
                                  range=[[clims[0,0]-0.5*(1/supercell_size), 
                                          clims[0,1]+0.5*(1/supercell_size)], 
                                         [clims[1,0]-0.5*(1/supercell_size), 
                                          clims[1,1]+0.5*(1/supercell_size)],
                                         [clims[2,0]-0.5*(1/supercell_size), 
                                          clims[2,1]+0.5*(1/supercell_size)]],
                                  expand_binnumbers=True).binnumber
            # validate the binning
            atom_indices = np.array([bin_indices[0,i]+(bin_indices[1,i]-1)*supercell_size+(bin_indices[2,i]-1)*supercell_size**2 for i in range(bin_indices.shape[1])])
            bincount = np.unique(atom_indices, return_counts=True)[1]
            if len(bincount) != supercell_size**3:
                raise TypeError("Incorrect number of bins. ")
                
            if max(bincount) != min(bincount):
                raise ValueError("Not all bins contain exactly the same number of atoms (1). ")
            
            C3denv = np.empty((3,supercell_size**2,supercell_size))
            for i in range(3):
                dims = [0,1,2]
                dims.remove(i)
                binx = bin_indices[dims,:]
                
                binned = [[] for _ in range(supercell_size**2)]
                for j in range(binx.shape[1]):
                    binned[(binx[0,j]-1)+supercell_size*(binx[1,j]-1)].append(j)
                binned = np.array(binned)

                
                for j in range(binned.shape[0]): # sort each bin in "i" direction coords
                    binned[j,:] = np.array([x for _, x in sorted(zip(cc[binned[j,:],i], binned[j,:]))]).reshape(1,supercell_size)
                
                C3denv[i,:] = binned
            
            C3denv = C3denv.astype(int)

            num_nn = math.ceil((supercell_size-1)/2)
                          
            arr = np.empty((num_nn,3,CN.shape[0],cc.shape[0]))
            arr[:] = np.nan
            for i in range(supercell_size**2):
                for at in range(supercell_size):
                    for dire in range(3):    
                        for nn in range(num_nn):
                            pos1 = at+(nn+1)
                            if pos1 > supercell_size-1:
                                pos1 -= supercell_size
                            temp = np.sum(np.multiply(CN[:,C3denv[dire,i,at],:],CN[:,C3denv[dire,i,pos1],:]),axis=1)
                            arr[nn,dire,:,i*supercell_size+at] = temp   
            
            if np.isnan(np.sum(arr)): raise TypeError("Some element missing in array.")
            
            MOCorr = arr
            self.MOCorr = MOCorr
            
            if read_mode == 2:
                Mobj = draw_MO_order_time(MOCorr, self.Ltimeline, uniname, saveFigures=False, smoother=smoother)
                self.Mobj = Mobj
                
            elif read_mode == 1:
                draw_MO_spatial_corr_NN12(MOCorr, uniname, saveFigures) 
                draw_MO_spatial_corr(MOCorr, uniname, saveFigures = False)
                MOspaC = draw_MO_spatial_corr_norm_var(MOCorr, uniname, saveFigures)
                self.MO_spatial_corr = MOspaC
                #self.prop_lib["MO_spatial_corr_length"] = MOspaC
                #print(f"MO spatial correlation length: \n {np.round(MOspaC,3)}")
            
            
        et1 = time.time()
        self.timing["MO"] = et1-et0
    
    
    def radial_distribution(self,allow_equil,uniname,saveFigures):
        """
        Radial distribution function (RDF) analysis.

        Parameters
        ----------

        """
        
        et0 = time.time()
        
        from pdyna.analysis import draw_RDF
        from pdyna.structural import get_volume
        
        trajnum = list(range(round(self.nframe*allow_equil),self.nframe,self.read_every))
        
        Xindex = self.Xindex
        Bindex = self.Bindex
        Cpos = self.Allpos[:,self.Cindex,:]
        Npos = self.Allpos[:,self.Nindex,:]
        Bpos = self.Allpos[:,Bindex,:]
        Xpos = self.Allpos[:,Xindex,:]
        
        CNtol = 2.7
        BXtol = 24
        
        if self._flag_organic_A:
            
            assert len(Xindex)/len(Bindex) == 3

            CNda = np.empty((0,))
            BXda = np.empty((0,))
            for i,fr in enumerate(trajnum):
                CNr=distance_matrix_handler(Cpos[fr,:],Npos[fr,:],self.latmat[fr,:],self.at0.cell,self.at0.pbc,self.complex_pbc)
                BXr=distance_matrix_handler(Bpos[fr,:],Xpos[fr,:],self.latmat[fr,:],self.at0.cell,self.at0.pbc,self.complex_pbc)

                CNda = np.concatenate((CNda,CNr[CNr<CNtol]),axis = 0)
                BXda = np.concatenate((BXda,BXr[BXr<BXtol]),axis = 0)

            draw_RDF(BXda, "BX", uniname, False)
            draw_RDF(CNda, "CN", uniname, False)
            ccn,bcn1 = np.histogram(CNda,bins=100,range=[1.38,1.65])
            bcn = 0.5*(bcn1[1:]+bcn1[:-1])
            cbx,bbx1 = np.histogram(BXda,bins=300,range=[0,5])
            bbx = 0.5*(bbx1[1:]+bbx1[:-1])
            BXRDF = bbx,cbx
            CNRDF = bcn,ccn
            
            self.BX_RDF = BXRDF
            self.BXbond = BXda
            self.CN_RDF = CNRDF
            self.CNbond = CNda
        
        else:
            
            assert len(Xindex)/len(Bindex) == 3

            BXda = np.empty((0,))
            for i,fr in enumerate(trajnum):
                BXr=distance_matrix_handler(Bpos[fr,:],Xpos[fr,:],self.latmat[fr,:],self.at0.cell,self.at0.pbc,self.complex_pbc)
                BXda = np.concatenate((BXda,BXr[BXr<BXtol]),axis = 0)

            draw_RDF(BXda, "BX", uniname, False)
            cbx,bbx1 = np.histogram(BXda,bins=300,range=[0,5])
            bbx = 0.5*(bbx1[1:]+bbx1[:-1])
            BXRDF = bbx,cbx
            
            self.BX_RDF = BXRDF
            self.BXbond = BXda
        
        et1 = time.time()
        self.timing["RDF"] = et1-et0
    

    def site_displacement(self,allow_equil,uniname,saveFigures):
        
        """
        A-site cation displacement analysis.

        Parameters
        ----------

        """

        from pdyna.structural import centmass_organic, centmass_organic_vec, find_B_cage_and_disp, get_frac_from_cart
        from pdyna.analysis import fit_3D_disp_atomwise, fit_3D_disp_total, peaks_3D_scatter
        
        et0 = time.time()
        
        st0 = self.st0
        st0pos = self.st0.cart_coords
        Allpos = self.Allpos
        latmat = self.latmat
        mymat = st0.lattice.matrix
        
        Bindex = self.Bindex
        Xindex = self.Xindex
        Hindex = self.Hindex
        
        
        # A-site displacements
        
        Afa = self.A_sites["FA"]
        Ama = self.A_sites["MA"]
        Aazr = self.A_sites["Azr"]
        Aindex_cs = self.A_sites["Cs"]
        
        ABsep = 8.2
        st0Bpos = st0pos[Bindex,:]
        
        CN_H_tol = 1.35
        
        Aindex_fa = []
        Aindex_ma = []
        Aindex_azr = []
        
        B8envs = {}
        
        if len(Afa) > 0:
            for i,env in enumerate(Afa):
                dm = distance_matrix_handler(st0pos[env[0]+env[1],:],st0pos[Hindex,:],mymat,self.at0.cell,self.at0.pbc,self.complex_pbc)
                Hs = sorted(list(np.argwhere(dm<CN_H_tol)[:,1]))
                Aindex_fa.append(env+[Hs])
                
                cent = centmass_organic(st0pos,st0.lattice.matrix,env+[Hs])
                ri=distance_matrix_handler(cent,st0Bpos,mymat,self.at0.cell,self.at0.pbc,self.complex_pbc)
                Bs = []
                for j in range(ri.shape[1]):
                    if ri[0,j] < ABsep:
                        Bs.append(Bindex[j])

                try:
                    assert len(Bs) == 8
                    B8envs[env[0][0]] = Bs
                except AssertionError: # can't find with threshold distance, try using nearest 8 atoms
                    cent = centmass_organic_vec(Allpos,latmat,env+[Hs])
                    ri = np.empty((Allpos.shape[0],len(Bindex)))
                    for fr in range(Allpos.shape[0]):
                        ri[fr,:,]=distance_matrix_handler(cent[fr,:],Allpos[fr,Bindex,:],self.latmat[fr,:],self.at0.cell,self.at0.pbc,self.complex_pbc)

                    ri = np.expand_dims(np.average(ri,axis=0),axis=0)
                    
                    Bs = []
                    for j in range(ri.shape[1]):
                        if ri[0,j] < ABsep:
                            Bs.append(Bindex[j])
                    assert len(Bs) == 8
                    B8envs[env[0][0]] = Bs
        
        if len(Ama) > 0:
            for i,env in enumerate(Ama):
                dm = distance_matrix_handler(st0pos[env[0]+env[1],:],st0pos[Hindex,:],mymat,self.at0.cell,self.at0.pbc,self.complex_pbc)
                Hs = sorted(list(np.argwhere(dm<CN_H_tol)[:,1]))
                Aindex_ma.append(env+[Hs])
                
                cent = centmass_organic(st0pos,st0.lattice.matrix,env+[Hs])
                ri=distance_matrix_handler(cent,st0Bpos,mymat,self.at0.cell,self.at0.pbc,self.complex_pbc)
                
                Bs = []
                for j in range(ri.shape[1]):
                    if ri[0,j] < ABsep:
                        Bs.append(Bindex[j])
                try:
                    assert len(Bs) == 8
                    B8envs[env[0][0]] = Bs
                except AssertionError: # can't find with threshold distance, try using nearest 8 atoms
                    cent = centmass_organic_vec(Allpos,latmat,env+[Hs])
                    ri = np.empty((Allpos.shape[0],len(Bindex)))
                    for fr in range(Allpos.shape[0]):
                        ri[fr,:,]=distance_matrix_handler(cent[fr,:],Allpos[fr,Bindex,:],self.latmat[fr,:],self.at0.cell,self.at0.pbc,self.complex_pbc)

                    ri = np.expand_dims(np.average(ri,axis=0),axis=0)
                    
                    Bs = []
                    for j in range(ri.shape[1]):
                        if ri[0,j] < ABsep:
                            Bs.append(Bindex[j])
                    assert len(Bs) == 8
                    B8envs[env[0][0]] = Bs
        
        if len(Aazr) > 0:
            for i,env in enumerate(Aazr):
                dm = distance_matrix_handler(st0pos[env[0]+env[1],:], st0pos[Hindex,:],mymat,self.at0.cell,self.at0.pbc,self.complex_pbc)
                Hs = sorted(list(np.argwhere(dm<CN_H_tol)[:,1]))
                Aindex_azr.append(env+[Hs])
                
                cent = centmass_organic(st0pos,st0.lattice.matrix,env+[Hs])
                ri=distance_matrix_handler(cent,st0Bpos,mymat,self.at0.cell,self.at0.pbc,self.complex_pbc)
                
                Bs = []
                for j in range(ri.shape[1]):
                    if ri[0,j] < ABsep:
                        Bs.append(Bindex[j])
                try:
                    assert len(Bs) == 8
                    B8envs[env[0][0]] = Bs
                except AssertionError: # can't find with threshold distance, try using nearest 8 atoms
                    cent = centmass_organic_vec(Allpos,latmat,env+[Hs])
                    ri = np.empty((Allpos.shape[0],len(Bindex)))
                    for fr in range(Allpos.shape[0]):
                        ri[fr,:,]=distance_matrix_handler(cent[fr,:],Allpos[fr,Bindex,:],self.latmat[fr,:],self.at0.cell,self.at0.pbc,self.complex_pbc)
                    ri = np.expand_dims(np.average(ri,axis=0),axis=0)
                    
                    Bs = []
                    for j in range(ri.shape[1]):
                        if ri[0,j] < ABsep:
                            Bs.append(Bindex[j])
                    assert len(Bs) == 8
                    B8envs[env[0]] = Bs        
        
                
        if len(Aindex_cs) > 0:
            for i,env in enumerate(Aindex_cs):
                ri=distance_matrix_handler(st0.cart_coords[env,:],st0Bpos,mymat,self.at0.cell,self.at0.pbc,self.complex_pbc)
                Bs = []
                for j in range(ri.shape[1]):
                    if ri[0,j] < ABsep:
                        Bs.append(Bindex[j])
                assert len(Bs) == 8
                B8envs[env] = Bs

    
        ranger = self.nframe
        ranger0 = round(ranger*allow_equil)
        
        Allposfr = Allpos[ranger0:,:,:]
        latmatfr = latmat[ranger0:,:]
        
        readTimestep = self.MDTimestep #*read_every

        if len(Aindex_ma) > 0:
            disp_ma = np.empty((Allposfr.shape[0],len(Aindex_ma),3))
            for ai, envs in enumerate(Aindex_ma):
                cent = centmass_organic_vec(Allposfr,latmatfr,envs)
                disp_ma[:,ai,:] = find_B_cage_and_disp(Allposfr,latmatfr,cent,B8envs[envs[0][0]])
            
            self.disp_ma = disp_ma
            dispvec_ma = disp_ma.reshape(-1,3)
            
            moltype = "MA"
            peaks_ma = fit_3D_disp_atomwise(disp_ma,readTimestep,uniname,moltype,saveFigures,title=moltype)
            fit_3D_disp_total(dispvec_ma,uniname,moltype,saveFigures,title=moltype)
            peaks_3D_scatter(peaks_ma,uniname,moltype,saveFigures)
            
        if len(Aindex_azr) > 0:
            disp_azr = np.empty((Allposfr.shape[0],len(Aindex_azr),3))
            for ai, envs in enumerate(Aindex_azr):
                cent = centmass_organic_vec(Allposfr,latmatfr,envs)
                disp_azr[:,ai,:] = find_B_cage_and_disp(Allposfr,latmatfr,cent,B8envs[envs[0][0]])
            
            self.disp_azr = disp_azr
            dispvec_azr = disp_azr.reshape(-1,3)
            
            moltype = "Azr"
            peaks_azr = fit_3D_disp_atomwise(disp_azr,readTimestep,uniname,moltype,saveFigures,title=moltype)
            fit_3D_disp_total(dispvec_azr,uniname,moltype,saveFigures,title=moltype)
            peaks_3D_scatter(peaks_azr,uniname,moltype,saveFigures)
            
        if len(Aindex_fa) > 0:
            disp_fa = np.empty((Allposfr.shape[0],len(Aindex_fa),3))
            for ai, envs in enumerate(Aindex_fa):
                cent = centmass_organic_vec(Allposfr,latmatfr,envs)
                disp_fa[:,ai,:] = find_B_cage_and_disp(Allposfr,latmatfr,cent,B8envs[envs[0][0]])
            
            self.disp_fa = disp_fa
            dispvec_fa = disp_fa.reshape(-1,3)    
            
            moltype = "FA"
            peaks_fa = fit_3D_disp_atomwise(disp_fa,readTimestep,uniname,moltype,saveFigures,title=moltype)
            fit_3D_disp_total(dispvec_fa,uniname,moltype,saveFigures,title=moltype)
            peaks_3D_scatter(peaks_fa,uniname,moltype,saveFigures)

        if len(Aindex_cs) > 0:
            disp_cs = np.empty((Allposfr.shape[0],len(Aindex_cs),3))
            for ai, ind in enumerate(Aindex_cs):
                cent = Allposfr[:,ind,:]
                disp_cs[:,ai,:] = find_B_cage_and_disp(Allposfr,latmatfr,cent,B8envs[ind])
            
            self.disp_cs = disp_cs
            dispvec_cs = disp_cs.reshape(-1,3)
            
            moltype = "Cs"
            peaks_cs = fit_3D_disp_atomwise(disp_cs,readTimestep,uniname,moltype,saveFigures,title=moltype)
            fit_3D_disp_total(dispvec_cs,uniname,moltype,saveFigures,title=moltype)
            peaks_3D_scatter(peaks_cs,uniname,moltype,saveFigures)
            
            
        
        # B-site displacement
        from pdyna.structural import apply_pbc_cart_vecs
        neigh_list = self.octahedra.astype(int)
        
        Bdisp = []
        for i,b in enumerate(Bindex):
            temp = Allposfr[:,[b],:] - Allposfr[:,np.array(Xindex)[list(neigh_list[i,:])],:]
            Bdisppbc = apply_pbc_cart_vecs(temp, latmatfr)
            
            disp = np.mean(Bdisppbc,axis=1)[:,np.newaxis,:]
            Bdisp.append(disp)
        
        Bdisp = np.concatenate(Bdisp,axis=1)
        
        # A-B displacement corr
        ABdisp_corrcoeff = {}
        if len(Aindex_fa) > 0:
            ad = []
            bd = []
            for ai, a in enumerate(Afa):
                b8 = [bi for bi, b in enumerate(Bindex) if b in B8envs[a[0][0]]]
                #np.multiply(disp_fa[:,[ai],:],Bdisp[:,b8,:])
                ad.append(disp_fa[:,[ai],:].reshape(-1,3))
                bd.append(np.mean(Bdisp[:,b8,:],axis=1).reshape(-1,3))
                
            ad = np.concatenate(ad,axis=0)
            bd = np.concatenate(bd,axis=0)
            cco = [np.corrcoef(ad[:,i],bd[:,i])[0,1] for i in range(3)]
            ABdisp_corrcoeff['FA'] = cco
            
        if len(Aindex_ma) > 0:
            ad = []
            bd = []
            for ai, a in enumerate(Ama):
                b8 = [bi for bi, b in enumerate(Bindex) if b in B8envs[a[0][0]]]
                #np.multiply(disp_fa[:,[ai],:],Bdisp[:,b8,:])
                ad.append(disp_ma[:,[ai],:].reshape(-1,3))
                bd.append(np.mean(Bdisp[:,b8,:],axis=1).reshape(-1,3))
                
            ad = np.concatenate(ad,axis=0)
            bd = np.concatenate(bd,axis=0)
            cco = [np.corrcoef(ad[:,i],bd[:,i])[0,1] for i in range(3)]
            ABdisp_corrcoeff['MA'] = cco
        
        if len(Aindex_azr) > 0:
            ad = []
            bd = []
            for ai, a in enumerate(Aazr):
                b8 = [bi for bi, b in enumerate(Bindex) if b in B8envs[a[0][0]]]
                #np.multiply(disp_fa[:,[ai],:],Bdisp[:,b8,:])
                ad.append(disp_azr[:,[ai],:].reshape(-1,3))
                bd.append(np.mean(Bdisp[:,b8,:],axis=1).reshape(-1,3))
                
            ad = np.concatenate(ad,axis=0)
            bd = np.concatenate(bd,axis=0)
            cco = [np.corrcoef(ad[:,i],bd[:,i])[0,1] for i in range(3)]
            ABdisp_corrcoeff['Azr'] = cco
            
        if len(Aindex_cs) > 0:
            ad = []
            bd = []
            for ai, a in enumerate(Aindex_cs):
                b8 = [bi for bi, b in enumerate(Bindex) if b in B8envs[a[0][0]]]
                #np.multiply(disp_fa[:,[ai],:],Bdisp[:,b8,:])
                ad.append(disp_cs[:,[ai],:].reshape(-1,3))
                bd.append(np.mean(Bdisp[:,b8,:],axis=1).reshape(-1,3))
                
            ad = np.concatenate(ad,axis=0)
            bd = np.concatenate(bd,axis=0)
            cco = [np.corrcoef(ad[:,i],bd[:,i])[0,1] for i in range(3)]
            ABdisp_corrcoeff['Cs'] = cco
        
        self.ABdisp_corrcoeff = ABdisp_corrcoeff
        self.prop_lib['AB_disp_corr'] = self.ABdisp_corrcoeff
        
        
        # B-B displacement corr
        import math
        from scipy.stats import binned_statistic_dd as binstat

        cc = self.st0.frac_coords[self.Bindex,:]
        
        supercell_size = self.supercell_size
        
        clims = np.array([[(np.quantile(cc[:,0],1/(supercell_size**2))+np.amin(cc[:,0]))/2,(np.quantile(cc[:,0],1-1/(supercell_size**2))+np.amax(cc[:,0]))/2],
                          [(np.quantile(cc[:,1],1/(supercell_size**2))+np.amin(cc[:,1]))/2,(np.quantile(cc[:,1],1-1/(supercell_size**2))+np.amax(cc[:,1]))/2],
                          [(np.quantile(cc[:,2],1/(supercell_size**2))+np.amin(cc[:,2]))/2,(np.quantile(cc[:,2],1-1/(supercell_size**2))+np.amax(cc[:,2]))/2]])
        
        bin_indices = binstat(cc, None, 'count', bins=[supercell_size,supercell_size,supercell_size], 
                              range=[[clims[0,0]-0.5*(1/supercell_size), 
                                      clims[0,1]+0.5*(1/supercell_size)], 
                                     [clims[1,0]-0.5*(1/supercell_size), 
                                      clims[1,1]+0.5*(1/supercell_size)],
                                     [clims[2,0]-0.5*(1/supercell_size), 
                                      clims[2,1]+0.5*(1/supercell_size)]],
                              expand_binnumbers=True).binnumber
        # validate the binning
        atom_indices = np.array([bin_indices[0,i]+(bin_indices[1,i]-1)*supercell_size+(bin_indices[2,i]-1)*supercell_size**2 for i in range(bin_indices.shape[1])])
        bincount = np.unique(atom_indices, return_counts=True)[1]
        if len(bincount) != supercell_size**3:
            raise TypeError("Incorrect number of bins. ")
        if max(bincount) != min(bincount):
            raise ValueError("Not all bins contain exactly the same number of atoms (1). ")
       
        bin_indices = bin_indices-1 # 0-indexing
        
        num_nn = math.ceil((supercell_size-1)/2)
        
        bb1 = []
        bb2 = []
        for o in range(bin_indices.shape[1]):
            si = bin_indices[:,[o]]
            b2d = []
            for space in range(3):
                addit = np.array([[0],[0],[0]])
                addit[space,0] = 1
                pos1 = si + addit
                pos1[pos1>supercell_size-1] = pos1[pos1>supercell_size-1]-supercell_size
                k1 = np.where(np.all(bin_indices==pos1,axis=0))[0][0]
                
                b2d.append(Bdisp[:,[k1],:])
            b1d = Bdisp[:,[o],:]
            b2d = np.concatenate(b2d,axis=1)
            bb1.append(b1d)
            bb2.append(b2d)
        bb1 = np.array(bb1)
        bb2 = np.array(bb2)
        
        BBdisp_corrcoeff = np.empty((3,3))
        for tax in range(3):
            for corrax in range(3):
                BBdisp_corrcoeff[corrax,tax] = np.corrcoef(bb1[:,:,0,tax].reshape(-1,),bb2[:,:,corrax,tax].reshape(-1,))[0,1]
        
        self.BBdisp_corrcoeff = BBdisp_corrcoeff
        self.prop_lib['BB_disp_corr'] = self.BBdisp_corrcoeff
        
        
        # A-A displacement corr
        MOcents=self.MOcenter
        if len(MOcents) == 1: # only run with pure A-site case
            MOcent = MOcents[list(MOcents.keys())[0]]
            if len(Aindex_fa) > 0:
                Adisp = self.disp_fa
            if len(Aindex_ma) > 0:
                Adisp = self.disp_ma
            if len(Aindex_cs) > 0:
                Adisp = self.disp_cs
            if len(Aindex_azr) > 0:
                Adisp = self.disp_azr
            
            safety_margin = np.array([1-1/supercell_size,1])
            
            cc = get_frac_from_cart(MOcent, latmatfr)[0,:]
            
            rect = [0,0,0]
            if np.argmin(np.abs(safety_margin - (np.amax(cc[:,0])-np.amin(cc[:,0])))): 
                cc[:,0] = cc[:,0]-(np.amax(cc[:,0])-(1-1/supercell_size/2))
                rect[0] = 1
            if np.argmin(np.abs(safety_margin - (np.amax(cc[:,1])-np.amin(cc[:,1])))): 
                cc[:,1] = cc[:,1]-(np.amax(cc[:,1])-(1-1/supercell_size/2))
                rect[1] = 1
            if np.argmin(np.abs(safety_margin - (np.amax(cc[:,2])-np.amin(cc[:,2])))): 
                cc[:,2] = cc[:,2]-(np.amax(cc[:,2])-(1-1/supercell_size/2))
                rect[2] = 1
            
            for i in range(3):
                if rect[i] == 0:
                    if np.amin(cc[:,i]) < 1/supercell_size/4:
                        cc[:,i] = cc[:,i] + 1/supercell_size/8*3
                    if np.amin(cc[:,i]) > 1-1/supercell_size/4:
                        cc[:,i] = cc[:,i] - 1/supercell_size/8*3
            
            for i in range(cc.shape[0]):
                for j in range(cc.shape[1]):
                    if cc[i,j] > 1:
                        cc[i,j] = cc[i,j]-1
                    if cc[i,j] < 0:
                        cc[i,j] = cc[i,j]+1
            
            clims = np.array([[(np.quantile(cc[:,0],1/(supercell_size**2))+np.amin(cc[:,0]))/2,(np.quantile(cc[:,0],1-1/(supercell_size**2))+np.amax(cc[:,0]))/2],
                              [(np.quantile(cc[:,1],1/(supercell_size**2))+np.amin(cc[:,1]))/2,(np.quantile(cc[:,1],1-1/(supercell_size**2))+np.amax(cc[:,1]))/2],
                              [(np.quantile(cc[:,2],1/(supercell_size**2))+np.amin(cc[:,2]))/2,(np.quantile(cc[:,2],1-1/(supercell_size**2))+np.amax(cc[:,2]))/2]])
            
            bin_indices = binstat(cc, None, 'count', bins=[supercell_size,supercell_size,supercell_size], 
                                  range=[[clims[0,0]-0.5*(1/supercell_size), 
                                          clims[0,1]+0.5*(1/supercell_size)], 
                                         [clims[1,0]-0.5*(1/supercell_size), 
                                          clims[1,1]+0.5*(1/supercell_size)],
                                         [clims[2,0]-0.5*(1/supercell_size), 
                                          clims[2,1]+0.5*(1/supercell_size)]],
                                  expand_binnumbers=True).binnumber
            # validate the binning
            atom_indices = np.array([bin_indices[0,i]+(bin_indices[1,i]-1)*supercell_size+(bin_indices[2,i]-1)*supercell_size**2 for i in range(bin_indices.shape[1])])
            bincount = np.unique(atom_indices, return_counts=True)[1]
            if len(bincount) != supercell_size**3:
                raise TypeError("Incorrect number of bins. ")
                
            if max(bincount) != min(bincount):
                raise ValueError("Not all bins contain exactly the same number of atoms (1). ")
           
            bin_indices = bin_indices-1 # 0-indexing
            
            num_nn = math.ceil((supercell_size-1)/2)
            
            aa1 = []
            aa2 = []
            for o in range(bin_indices.shape[1]):
                si = bin_indices[:,[o]]
                a2d = []
                for space in range(3):
                    addit = np.array([[0],[0],[0]])
                    addit[space,0] = 1
                    pos1 = si + addit
                    pos1[pos1>supercell_size-1] = pos1[pos1>supercell_size-1]-supercell_size
                    k1 = np.where(np.all(bin_indices==pos1,axis=0))[0][0]
                    
                    a2d.append(Adisp[:,[k1],:])
                a1d = Adisp[:,[o],:]
                a2d = np.concatenate(a2d,axis=1)
                aa1.append(a1d)
                aa2.append(a2d)
            aa1 = np.array(aa1)
            aa2 = np.array(aa2)
            
            AAdisp_corrcoeff = np.empty((3,3))
            for tax in range(3):
                for corrax in range(3):
                    AAdisp_corrcoeff[corrax,tax] = np.corrcoef(aa1[:,:,0,tax].reshape(-1,),aa2[:,:,corrax,tax].reshape(-1,))[0,1]
            
            self.AAdisp_corrcoeff = AAdisp_corrcoeff
            self.prop_lib['AA_disp_corr'] = self.AAdisp_corrcoeff
        
        
        et1 = time.time()
        self.timing["A_disp"] = et1-et0
    
    
    def property_processing(self,allow_equil,read_mode,octa_locality,uniname,saveFigures):
        
        """
        Post-processing of computed properties.
        
        Parameters
        ----------

        octa_locality : compute differentiated properties within mixed-halide sample
            - configuration of each octahedron, giving 10 types according to halide geometry
            - quantify Br- and I-rich regions with concentration
        """
        
        et0 = time.time()
        
        if octa_locality:
            from pdyna.structural import octahedra_coords_into_bond_vectors, match_mixed_halide_octa_dot
            
            st0 = self.st0
            Bindex = self.Bindex
            Xindex = self.Xindex
            Bpos = self.Allpos[:,self.Bindex,:]
            Xpos = self.Allpos[:,self.Xindex,:]
            neigh_list = self.octahedra
            
            mymat = st0.lattice.matrix
            
            b0 = st0.cart_coords[Bindex,:]
            x0 = st0.cart_coords[Xindex,:]
            
            Xspec = []
            Xonehot = np.empty((0,2))
            halcounts = np.zeros((2,)).astype(int)
            for i,site in enumerate([st0.sites[i] for i in Xindex]): 
                 if site.species_string == 'Br':
                     halcounts[1] = halcounts[1] + 1
                     Xspec.append("Br")  
                     Xonehot = np.concatenate((Xonehot,np.array([[0,1]])),axis=0)
                 elif site.species_string == 'I':
                     halcounts[0] = halcounts[0] + 1
                     Xspec.append("I")  
                     Xonehot = np.concatenate((Xonehot,np.array([[1,0]])),axis=0)
                 else:
                     raise TypeError(f"A X-site element {site.species_string} is found other than I and Br. ")
            
            r = distance_matrix_handler(b0,x0,mymat,self.at0.cell,self.at0.pbc,self.complex_pbc)
            
            # compare with final config to make sure there is no change of env
            b1 = Bpos[-1,:]
            x1 = Xpos[-1,:]
            rf = distance_matrix_handler(b1,x1,self.latmat[-1,:],self.at0.cell,self.at0.pbc,self.complex_pbc)

            if np.amax(r-rf) > 3.5: 
                print(f"Warning: The maximum atomic position difference between initial and final configs are too large ({round(np.amax(r-rf),3)} A). ")
            
            octa_halide_code = [] # resolve the halides of a octahedron, key output
            octa_halide_code_single = []
            for B_site, X_list in enumerate(r): # for each B-site atom  
                
                raw = x0[neigh_list[B_site,:].astype(int),:] - b0[B_site,:]
                bx = octahedra_coords_into_bond_vectors(raw,mymat)
                
                hals = []
                for j in range(6):
                    hals.append(Xspec[int(neigh_list[B_site,j])])
                
                form_factor, ff_single = match_mixed_halide_octa_dot(bx,hals)
                octa_halide_code.append(form_factor) # determine each octa as one of the 10 configs, key output
                octa_halide_code_single.append(ff_single)
                
            # resolve local env of octahedron
            env_BX_distance = 10.7 # to cover approx. the third NN halides
            #plt.scatter(r.reshape(-1,),r.reshape(-1,))
            #plt.axvline(x=10.7)
            
            sampling = 8
  
            syscode = np.empty((0,len(Bindex),Xonehot.shape[1]))
            frread = list(np.round(np.linspace(round(Bpos.shape[0]*allow_equil),Bpos.shape[0]-1,sampling)).astype(int))
            
            for fr in frread:
                r = distance_matrix_handler(Bpos[fr,:],Xpos[fr,:],self.latmat[fr,:],self.at0.cell,self.at0.pbc,self.complex_pbc)
                Xcodemaster = np.empty((len(Bindex),Xonehot.shape[1]))
                for B_site, X_list in enumerate(r): # for each B-site atom
                    Xcoeff = 1/np.power(X_list,1.2)
                    Xcode = Xonehot.copy()
                    Xcode = Xcode[Xcoeff>(1/env_BX_distance),:]
                    Xcoeff = Xcoeff[Xcoeff>(1/env_BX_distance)]
                    Xcoeff = (np.array(Xcoeff)/sum(Xcoeff)).reshape(1,-1)
                    Xcodemaster[B_site,:] = np.sum(np.multiply(np.transpose(Xcoeff),Xcode),axis=0)
                    
                syscode = np.concatenate((syscode,np.expand_dims(Xcodemaster,axis=0)),axis=0)
        
            syscode = np.average(syscode,axis=0) # label each B atom with its neighbouring halide density in one-hot manner, key output
            brconc = syscode[:,1]

            # categorize the octa into different configs
            config_types = set(octa_halide_code_single)
            typelib = [[] for numm in range(len(config_types))]
            for ti, typei in enumerate(config_types):
                typelib[ti] = [k for k, x in enumerate(octa_halide_code_single) if x == typei]
            
            # activate local Br content analysis only if the sample is large enough
            brrange = [np.amin(syscode[:,1]),np.amax(syscode[:,1])]
            brbinnum = 12 # key parameter
            #diffbin = (brrange[1]-brrange[0])/brbinnum*0.5
            #binrange = [np.amin(syscode[:,1])-diffbin,np.amax(syscode[:,1])+diffbin]
            if syscode.shape[0] >= 64 and brrange[1]-brrange[0] > 0.12: 
                #Bins = np.linspace(binrange[0],binrange[1],brbinnum+1)
                qus = np.linspace(0,1,brbinnum+1)
                Bins = []
                for q in qus:
                    Bins.append(np.quantile(syscode[:,1],q))
                Bins = np.array(Bins)
                deltab = 0.0001
                Bins[0] = Bins[0] - deltab
                Bins[-1] = Bins[-1] + deltab
                bininds = np.digitize(syscode[:,1],Bins)-1
                brbins = [[] for kk in range(brbinnum)] # global index of B atoms that are in each bin of Bins
                for ibin, binnum in enumerate(bininds):
                    brbins[binnum].append(ibin)
                    
                bincent = (Bins[1:]+Bins[:-1])/2
            
            # partition properties
            if hasattr(self,"Tilting"):
                Di = self.Distortion
                T = self.Tilting
                TCNtype = []
                TCNconc = []
                if hasattr(self,"Tilting_Corr"):
                    from pdyna.analysis import get_tcp_from_list
                    TCN = self.Tilting_Corr
                    #TCN = get_norm_corr(TCN,T)
                    
                from pdyna.analysis import draw_octatype_tilt_density, draw_octatype_dist_density, draw_halideconc_tilt_density, draw_halideconc_dist_density, draw_halideconc_lat_density, draw_octatype_lat_density
                
                Dtype = []
                Ttype = []
                for ti, types in enumerate(typelib):
                    Dtype.append(Di[:,types,:])
                    Ttype.append(T[:,types,:])
                if hasattr(self,"Tilting_Corr"):
                    for ti, types in enumerate(typelib):
                        TCNtype.append(TCN[:,types,:])
                    tcptype = get_tcp_from_list(TCNtype)
                if len(TCNtype) == 0:
                    tcptype = None
                
                Tmaxs_type = draw_octatype_tilt_density(Ttype, typelib, config_types, uniname, saveFigures, corr_vals=tcptype)
                Dgauss_type, Dgaussstd_type = draw_octatype_dist_density(Dtype, config_types, uniname, saveFigures)
                
                concent = [] # concentrations recorded
                Dconc = []
                Tconc = []
                for ii,item in enumerate(brbins):
                    if len(item) == 0: continue
                    concent.append(bincent[ii])
                    Dconc.append(Di[:,item,:])
                    Tconc.append(T[:,item,:])
                if hasattr(self,"Tilting_Corr"):
                    for ii,item in enumerate(brbins):
                        TCNconc.append(TCN[:,item,:])
                    tcoconc = get_tcp_from_list(TCNconc)
                if len(TCNconc) == 0:
                    tcoconc = None
                
                Tmaxs_conc = draw_halideconc_tilt_density(Tconc, brconc, concent, uniname, saveFigures)
                Dgauss_conc, Dgaussstd_conc = draw_halideconc_dist_density(Dconc, concent, uniname, saveFigures)
                
                self.tilt_wrt_halideconc = [concent,Tmaxs_conc]
                self.dist_wrt_halideconc = [concent,Dgauss_conc]
                
                if read_mode == 1:
                    self.prop_lib['distortion_halideconc'] = self.dist_wrt_halideconc
                    self.prop_lib['tilting_halideconc'] = self.tilt_wrt_halideconc
                
                
                
                # partition tilting correlation length wrt local config
                if hasattr(self,"spatialCorrLength"):
                    TC = self.spatialCorr['raw']
                    TC = np.sqrt(np.abs(TC))*np.sign(TC)

                    from pdyna.analysis import quantify_octatype_tilt_domain, quantify_halideconc_tilt_domain
                    
                    TCtype = []
                    for ti, types in enumerate(typelib):
                        temp = TC[types,:]
                        temp = temp/temp[:,:,[0],:]
                        TCtype.append(temp)
                    
                    Tmaxs_type = quantify_octatype_tilt_domain(TCtype, config_types, uniname, saveFigures)
                    
                    concent = [] # concentrations recorded
                    TCconc = []
                    for ii,item in enumerate(brbins):
                        if len(item) == 0: continue
                        concent.append(bincent[ii])
                        temp = TC[item,:]
                        temp = temp/temp[:,:,[0],:]
                        TCconc.append(temp)
                    
                    Tmaxs_conc = quantify_halideconc_tilt_domain(TCconc, concent, uniname, saveFigures)

                
            if hasattr(self,"Lat") and self.Lat.ndim == 3: #partition lattice parameter as well
                L = self.Lat
                Ltype = []
                for ti, types in enumerate(typelib):
                    Ltype.append(L[:,types,:])
                
                concent = [] # concentrations recorded
                Lconc = []
                for ii,item in enumerate(brbins):
                    if len(item) == 0: continue
                    concent.append(bincent[ii])
                    Lconc.append(L[:,item,:])
                    
                Lgauss_conc, Lgaussstd_conc = draw_halideconc_lat_density(Lconc, concent, uniname, saveFigures)
                Lgauss_type, Lgaussstd_type = draw_octatype_lat_density(Ltype, config_types, uniname, saveFigures)

                self.lat_wrt_halideconc = [concent,Lgauss_conc]
                if read_mode == 1:
                    self.prop_lib['lattice_halideconc'] = self.lat_wrt_halideconc
                
            # get distribution of partitioning
            from pdyna.analysis import print_partition
            #brrange = [np.amin(syscode[:,1]),np.amax(syscode[:,1])]
            #diffbin = (brrange[1]-brrange[0])/brbinnum*0.5
            #binrange = [np.amin(syscode[:,1])-diffbin,np.amax(syscode[:,1])+diffbin]
            #Bins = np.linspace(binrange[0],binrange[1],brbinnum+1)
            #bininds = np.digitize(syscode[:,1],Bins)-1
            #brbins = [[] for kk in range(brbinnum)]
            #for ibin, binnum in enumerate(bininds):
            #    brbins[binnum].append(ibin)
            print_partition(typelib,config_types,brconc,halcounts)
            
        et1 = time.time()
        self.timing["property_processing"] = et1-et0


@dataclass
class Frame:
    
    """
    Class for analysis of single-frame structure.
    Initialize the class with reading the raw data.

    Parameters
    ----------
    data_format : data format based on the software
        Currently compatible format is 'poscar'.
    data_path : tuple of input files
        The input file path.
        poscar: poscar_path

    """
    
    data_format: str = field(repr=False)
    data_path: str = field(repr=False)
    
    # characteristic value of bond length of your material for structure construction 
    # the first interval should be large enough to cover all the first and second NN of B-X (B-B) pairs, 
    # in the second list, the two elements are 0) approximate first NN distance of B-X (B-B) pairs, and 1) approximate second NN distance of B-X (B-B) pairs
    # These values can be obtained by inspecting the initial configuration or, e.g. in the pair distrition function of the structure
    #_fpg_val_BB = [[3,9.6], [6,8.8]] # empirical values for lead halide perovskites
    #_fpg_val_BX = [[0.1,8], [3,6.8]] # empirical values for lead halide perovskites
    _fpg_val_BB = [[4,9.1], [5.8,8.1]] # empirical values for lead halide perovskites
    _fpg_val_BX = [[2,7.5], [3,6.2]] # empirical values for lead halide perovskites
     
    def __post_init__(self):
        
        Xsite_species = ['Cl','Br','I'] # update if needed
        Bsite_species = ['Pb','Sn'] # update if needed
        known_elem = ("I", "Br", "Cl", "Pb", "C", "H", "N", "Cs") # update if needed
        
        #Xsite_species = ['Ba'] # update if needed
        #Bsite_species = ['P'] # update if needed
        #known_elem = ('Ba', 'P', 'Sb') # update if needed
        
        et0 = time.time()

        if self.data_format == 'poscar':
            
            import pymatgen.io.vasp.inputs as vi
            from pdyna.io import chemical_from_formula
            
            poscar_path = self.data_path
            
            print("------------------------------------------------------------")
            print("Loading Frame files...")
            
            # read POSCAR and XDATCAR files
            st0 = vi.Poscar.from_file(poscar_path,check_for_POTCAR=False).structure # initial configuration
            
            for elem in st0.symbol_set:
                if not elem in known_elem:
                    raise ValueError(f"An unexpected element {elem} is found. Please check the list known_elem. ")
            self.st0 = st0
            at0 = aaa.get_atoms(st0)
            self.at0 = at0
            self.natom = len(st0)
            self.species_set = st0.symbol_set
            self.formula = chemical_from_formula(st0)
            self.scaled_up = False

            # read the coordinates and save   
            Xindex = []
            Bindex = []

            for i,site in enumerate(st0.sites):
                 if site.species_string in Xsite_species:
                     Xindex.append(i)
                 if site.species_string in Bsite_species:
                     Bindex.append(i)  
            
            if len(Bindex) < 16:
                st0.make_supercell([2,2,2])
                
                self.st0 = st0
                self.natom = len(st0)
                self.scaled_up = True
                
                Xindex = []
                Bindex = []

                for i,site in enumerate(st0.sites):
                     if site.species_string in Xsite_species:
                         Xindex.append(i)
                     if site.species_string in Bsite_species:
                         Bindex.append(i)  
            
            self.Bindex = Bindex
            self.Xindex = Xindex
            
        elif self.data_format == 'cif':
            
            from ase.io import read
            from pdyna.io import chemical_from_formula
            
            poscar_path = self.data_path
            
            print("------------------------------------------------------------")
            print("Loading Frame files...")
            
            # read files
            a=read(filename=poscar_path,format='cif')
            st0 = aaa.get_structure(a)
            
            for elem in st0.symbol_set:
                if not elem in known_elem:
                    raise ValueError(f"An unexpected element {elem} is found. Please check the list known_elem. ")
            self.st0 = st0
            at0 = aaa.get_atoms(st0)
            self.at0 = at0
            self.natom = len(st0)
            self.species_set = st0.symbol_set
            self.formula = chemical_from_formula(st0)
            self.scaled_up = False

            # read the coordinates and save   
            Xindex = []
            Bindex = []

            for i,site in enumerate(st0.sites):
                 if site.species_string in Xsite_species:
                     Xindex.append(i)
                 if site.species_string in Bsite_species:
                     Bindex.append(i)  
            
            if len(Bindex) < 16:
                st0.make_supercell([2,2,2])
                
                self.st0 = st0
                self.natom = len(st0)
                self.scaled_up = True
                
                Xindex = []
                Bindex = []

                for i,site in enumerate(st0.sites):
                     if site.species_string in Xsite_species:
                         Xindex.append(i)
                     if site.species_string in Bsite_species:
                         Bindex.append(i)  
            
            self.Bindex = Bindex
            self.Xindex = Xindex
            
        
        else:
            raise TypeError("Unsupported data format: {}".format(self.data_format))
        
        et1 = time.time()
        self.timing = {}
        self.timing["reading"] = et1-et0


    def __str__(self):
        pattern = '''
        Perovskite Frame
        Formula:          {}
        Number of atoms:  {}
        '''
        return pattern.format(self.formula, self.natom)
    
    
    def __repr__(self):
        return 'PDynA Frame({}, {} atoms)'.format(self.formula, self.natom)    
    
    
    def analysis(self,
                 # general parameters
                 uniname = "test", # A unique user-defined name for this trajectory, will be used in printing and figure saving
                 saveFigures = False, # whether to save produced figures
                 align_rotation = [0,0,0], # rotation of structure to match orthogonal directions
                 
                 tilt_corr_spatial = False,
                 ):
        
        """
        Core function for analysing perovskite trajectory.
        The parameters are used to enable various analysis functions and handle their functionality.

        Parameters
        ----------
        
        -- General Parameters
        uniname     -- unique user-defined name for this trajectory, will be used in printing and figure saving
        
        -- Saving the Outputs
        saveFigures   -- whether to save produced figures
            True or False

        -- Octahedral Tilting and Distortion
        align_rotation  -- rotation angles about [a,b,c] in angles 
        p.s. The 'True or False' options all have False as the default unless specified otherwise. 
        """
        
        # pre-definitions
        print("Current sample:",uniname)
        #print("Initializing trajectory")
        
        # reset timing
        self.timing = {"reading": self.timing["reading"]}
        self.uniname = uniname
        
        et0 = time.time()
        print(" ")
        
        # label the constituent octahedra
        from pdyna.structural import fit_octahedral_network_frame
        
        st0Bpos = self.st0.cart_coords[self.Bindex,:]
        st0Xpos = self.st0.cart_coords[self.Xindex,:]
        mymat = self.st0.lattice.matrix
        
        rotated = False
        rotmat = None
        if align_rotation != [0,0,0]: 
            rotated = True
            from scipy.spatial.transform import Rotation as sstr
            align_rotation = np.array(align_rotation)/180*np.pi
            rotmat = sstr.from_rotvec(align_rotation).as_matrix().reshape(3,3)
        
        at0 = aaa.get_atoms(self.st0)
        self.at0 = at0
        angles = self.st0.lattice.angles
        
        if (max(angles) < 100 and min(angles) > 80):
            self.complex_pbc = False
            rt=distance_matrix_handler(st0Bpos,st0Xpos,mymat)
        else:
            self.complex_pbc = True
            rt=distance_matrix_handler(st0Bpos,st0Xpos,mymat,at0.cell,at0.pbc,True)
        
        if rotated:
            neigh_list = fit_octahedral_network_frame(st0Bpos,st0Xpos,rt,mymat,self._fpg_val_BX,rotated,rotmat)
        else:
            neigh_list = fit_octahedral_network_frame(st0Bpos,st0Xpos,rt,mymat,self._fpg_val_BX,rotated,None)
        
        self.rotated = rotated
        self.frame_rotation = rotmat
        self.octahedra = neigh_list
            
        et1 = time.time()
        self.timing["env_resolve"] = et1-et0
        
        
        print("Computing octahedral tilting and distortion...")
        self.tilting_and_distortion(uniname=uniname,saveFigures=saveFigures,tilt_corr_spatial=tilt_corr_spatial)
        print(" ")

        # summary
        self.timing["total"] = sum(list(self.timing.values()))
        print(" ")
        print_time(self.timing)
        
        # end of calculation
        print("------------------------------------------------------------")
    
    def tilting_and_distortion(self,uniname,saveFigures,tilt_corr_spatial):
        
        """
        Octhedral tilting and distribution analysis.

        """
        
        from scipy.spatial.transform import Rotation as sstr
        from pdyna.structural import octahedra_coords_into_bond_vectors, calc_distortions_from_bond_vectors
        from pdyna.analysis import draw_dist_density_frame
        
        et0 = time.time()
        
        mymat = self.st0.lattice.matrix
        Bcount = len(self.Bindex)
        neigh_list = self.octahedra
        Bpos = self.st0.cart_coords[self.Bindex,:]
        Xpos = self.st0.cart_coords[self.Xindex,:]
        if self.rotated:
            rotmat = np.linalg.inv(self.frame_rotation)
        
        D = np.empty((0,4))
        Rmat = np.zeros((Bcount,3,3))
        Rmsd = np.zeros((Bcount,1))
        for B_site in range(Bcount): # for each B-site atom
            raw = Xpos[neigh_list[B_site,:].astype(int),:] - Bpos[B_site,:]
            bx = octahedra_coords_into_bond_vectors(raw,mymat)
            if self.rotated:
                bx = np.matmul(bx,rotmat)
      
            dist_val,rot,rmsd = calc_distortions_from_bond_vectors(bx)
                
            Rmat[B_site,:] = rot
            Rmsd[B_site] = rmsd
            D = np.concatenate((D,dist_val.reshape(1,4)),axis = 0)
                
        
        T = np.zeros((Bcount,3))
        for i in range(Rmat.shape[0]):
            T[i,:] = sstr.from_matrix(Rmat[i,:]).as_euler('xyz', degrees=True)
        
        self.Tilting = T
        self.Distortion = D
        
        dmu = draw_dist_density_frame(D, uniname, saveFigures, n_bins = 100)        
        print("Octahedral distortions:",np.round(dmu,4))
        
        
        # NN1 correlation function of tilting (Glazer notation)
        from pdyna.analysis import abs_sqrt, draw_tilt_and_corr_density_shade_frame
        from pdyna.structural import find_population_gap, apply_pbc_cart_vecs_single_frame
        
        #default_BB_dist = 6.1
        r0=distance_matrix_handler(self.st0.cart_coords[self.Bindex,:],self.st0.cart_coords[self.Bindex,:],mymat,self.at0.cell,self.at0.pbc,self.complex_pbc)
        
        try: # the old way
            search_NN1 = find_population_gap(r0, self._fpg_val_BB[0], self._fpg_val_BB[1])
            default_BB_dist = np.mean(r0[np.logical_and(r0>0.1,r0<search_NN1)])
            
            Benv = []
            for B1, B2_list in enumerate(r0): # find the nearest Pb within a cutoff
                Benv.append([i for i,B2 in enumerate(B2_list) if (B2 > 0.1 and B2 < search_NN1)])
            Benv = np.array(Benv)
            
            aa = Benv.shape[1] # if some of the rows in Benv don't have 6 neighbours.
    
        except (IndexError, ValueError) as e: # solve env through 3D binning
            print("!Normal fitting of initial B-B structure failed (B sites are displaced too much), trying a secondary binning fit. ")
            cell_lat = np.array(self.st0.lattice.abc)[np.newaxis,:]
            cell_diff = np.amax(np.abs(np.repeat(cell_lat,3,axis=0)-np.repeat(cell_lat.T,3,axis=1)))
            #cubic_cell = True
            if cell_diff < 2.5:
                import math
                from scipy.stats import binned_statistic_dd as binstat
                from pdyna.structural import apply_pbc_cart_vecs_single_frame
                from pdyna.analysis import quantify_tilt_domain

                cc = self.st0.frac_coords[self.Bindex,:]
                if not np.cbrt(len(self.Bindex)).is_integer():
                    raise ValueError(f"The number of B sites is {len(self.Bindex)}, which is not a cubed number, indicating a non-cubic cell. ")
                    
                supercell_size = int(np.cbrt(len(self.Bindex)))
                
                for i in range(3):
                    if abs(np.amax(cc[:,i])-1)<0.01 or abs(np.amin(cc[:,i]))<0.01:
                        addit = np.zeros((1,3))
                        addit[0,i] = 1/supercell_size/2
                        cc = cc+addit
                        
                cc[cc>1] = cc[cc>1]-1
                    
                clims = np.array([[(np.quantile(cc[:,0],1/(supercell_size**2))+np.amin(cc[:,0]))/2,(np.quantile(cc[:,0],1-1/(supercell_size**2))+np.amax(cc[:,0]))/2],
                                  [(np.quantile(cc[:,1],1/(supercell_size**2))+np.amin(cc[:,1]))/2,(np.quantile(cc[:,1],1-1/(supercell_size**2))+np.amax(cc[:,1]))/2],
                                  [(np.quantile(cc[:,2],1/(supercell_size**2))+np.amin(cc[:,2]))/2,(np.quantile(cc[:,2],1-1/(supercell_size**2))+np.amax(cc[:,2]))/2]])
                
                bin_indices = binstat(cc, None, 'count', bins=[supercell_size,supercell_size,supercell_size], 
                                      range=[[clims[0,0]-0.5*(1/supercell_size), 
                                              clims[0,1]+0.5*(1/supercell_size)], 
                                             [clims[1,0]-0.5*(1/supercell_size), 
                                              clims[1,1]+0.5*(1/supercell_size)],
                                             [clims[2,0]-0.5*(1/supercell_size), 
                                              clims[2,1]+0.5*(1/supercell_size)]],
                                      expand_binnumbers=True).binnumber
                # validate the binning
                atom_indices = np.array([bin_indices[0,i]+(bin_indices[1,i]-1)*supercell_size+(bin_indices[2,i]-1)*supercell_size**2 for i in range(bin_indices.shape[1])])
                bincount = np.unique(atom_indices, return_counts=True)[1]
                if len(bincount) != supercell_size**3:
                    raise ValueError("Incorrect number of bins. ")
                if max(bincount) != min(bincount):
                    raise ValueError("Not all bins contain exactly the same number of atoms (1). ")
                
                Benv = []
                indmap = np.array([[1,0,0],[0,1,0],[0,0,1],[-1,0,0],[0,-1,0],[0,0,-1]])
                for B1 in range(bin_indices.shape[1]):
                    temp = bin_indices[:,[B1]].T+indmap
                    temp[temp>supercell_size] = temp[temp>supercell_size] - supercell_size
                    temp[temp<1] = temp[temp<1] + supercell_size
                    kt = []
                    for j in range(6):
                        kt.append(np.where(np.all(bin_indices==temp[[j],:].T,axis=0))[0][0])
                    kt = sorted(kt)
                    Benv.append(kt)
                    
                Benv = np.array(Benv)
                
                Bpos = self.st0.cart_coords[self.Bindex,:]
                
                bbvecs = []
                for o in range(Benv.shape[0]):
                    bbvecs.append(Bpos[[o],:]-Bpos[Benv[o,:],:])
                bv = np.concatenate(bbvecs,axis=0)
                default_BB_dist = np.mean(np.linalg.norm(apply_pbc_cart_vecs_single_frame(bv, mymat),axis=1))
                
            
            else:
                raise ValueError("The cell is not in cubic shape. ")
        

# =============================================================================
#             except ValueError:
#                 print(f"Need to adjust the range of B atom 1st NN distance (was {search_NN1}).  ")
#                 print("See the gap between the populations. \n")
#                 test_range = r0.reshape((1,r0.shape[0]**2))
#                 import matplotlib.pyplot as plt
#                 fig,ax = plt.subplots(1,1)
#                 plt.hist(test_range.reshape(-1,),range=[5,9],bins=100)
#             except IndexError:
#                 print(f"Need to adjust the range of B atom 1st NN distance (was {search_NN1}).  ")
#                 print("See the gap between the populations. \n")
#                 test_range = r0.reshape((1,r0.shape[0]**2))
#                 import matplotlib.pyplot as plt
#                 fig,ax = plt.subplots(1,1)
#                 plt.hist(test_range.reshape(-1,),range=[5,9],bins=100)
# =============================================================================
                    
            
        cubic_cell = False    
        if Benv.shape[1] == 3: # indicate a 2*2*2 supercell
            
            raise ValueError("The cell size is too small, PBC can't be analyzed. ")

        elif Benv.shape[1] == 6: # indicate a larger supercell
            cubic_cell = True
            Bcoordenv = np.empty((Benv.shape[0],6,3))
            for i in range(Benv.shape[0]):
                Bcoordenv[i,:] = Bpos[Benv[i,:],:] - Bpos[i,:]
            
            Bcoordenv = apply_pbc_cart_vecs_single_frame(Bcoordenv,mymat)    
                            
            ref_octa = np.array([[1,0,0],[-1,0,0],
                                 [0,1,0],[0,-1,0],
                                 [0,0,1],[0,0,-1]])
            for i in range(Bcoordenv.shape[0]):
                orders = np.zeros((1,6))
                for j in range(6):
                    temp = Bcoordenv[i,:,:]
                    if self.rotated:
                        temp = np.matmul(temp,rotmat)
                    orders[0,j] = np.argmax(np.dot(temp,ref_octa[j,:]))
                Benv[i,:] = Benv[i,:][orders.astype(int)]
                    
            # now each row of Benv contains the Pb atom index that sit in x,y and z direction of the row-numbered Pb atom.
            Corr = np.empty((T.shape[0],6))
            for B1 in range(T.shape[0]):
                    
                Corr[B1,[0,1]] = abs_sqrt(T[[B1],0]*T[Benv[B1,[0,1]],0]) # x neighbour 1,2
                Corr[B1,[2,3]] = abs_sqrt(T[[B1],1]*T[Benv[B1,[2,3]],1]) # y neighbour 1,2
                Corr[B1,[4,5]] = abs_sqrt(T[[B1],2]*T[Benv[B1,[4,5]],2]) # z neighbour 1,2
            
        else: 
            raise TypeError(f"The environment matrix is incorrect. {Benv.shape[1]} ")
        
        self._BNNenv = Benv
        self.Tilting_Corr = Corr
        
        tquant, polarity = draw_tilt_and_corr_density_shade_frame(T, Corr, uniname, saveFigures)
        #print("Tilt angles found:")
        #print(f"a-axis: {tquant[0]}")
        #print(f"b-axis: {tquant[1]}")
        #print(f"c-axis: {tquant[2]}")
        print("tilting correlation:",np.round(np.array(polarity).reshape(3,),3))
        
        self.tilt_peaks = tquant
        self.tilt_corr_polarity = polarity
        
        cell_lat = np.array(self.st0.lattice.abc)[np.newaxis,:]
        cell_diff = np.amax(np.abs(np.repeat(cell_lat,3,axis=0)-np.repeat(cell_lat.T,3,axis=1)))
        #cubic_cell = True
        
        if tilt_corr_spatial and cubic_cell and cell_diff < 2.5:
            import math
            from scipy.stats import binned_statistic_dd as binstat
            from pdyna.analysis import quantify_tilt_domain

            cc = self.st0.frac_coords[self.Bindex,:]
            supercell_size = round(np.mean(cell_lat)/default_BB_dist)
            for i in range(3):
                if abs(np.amax(cc[:,i])-1)<0.01 or abs(np.amin(cc[:,i]))<0.01:
                    addit = np.zeros((1,3))
                    addit[0,i] = 1/supercell_size/2
                    cc = cc+addit
                    
            cc[cc>1] = cc[cc>1]-1
                
            clims = np.array([[(np.quantile(cc[:,0],1/(supercell_size**2))+np.amin(cc[:,0]))/2,(np.quantile(cc[:,0],1-1/(supercell_size**2))+np.amax(cc[:,0]))/2],
                              [(np.quantile(cc[:,1],1/(supercell_size**2))+np.amin(cc[:,1]))/2,(np.quantile(cc[:,1],1-1/(supercell_size**2))+np.amax(cc[:,1]))/2],
                              [(np.quantile(cc[:,2],1/(supercell_size**2))+np.amin(cc[:,2]))/2,(np.quantile(cc[:,2],1-1/(supercell_size**2))+np.amax(cc[:,2]))/2]])
            
            bin_indices = binstat(cc, None, 'count', bins=[supercell_size,supercell_size,supercell_size], 
                                  range=[[clims[0,0]-0.5*(1/supercell_size), 
                                          clims[0,1]+0.5*(1/supercell_size)], 
                                         [clims[1,0]-0.5*(1/supercell_size), 
                                          clims[1,1]+0.5*(1/supercell_size)],
                                         [clims[2,0]-0.5*(1/supercell_size), 
                                          clims[2,1]+0.5*(1/supercell_size)]],
                                  expand_binnumbers=True).binnumber
            # validate the binning
            atom_indices = np.array([bin_indices[0,i]+(bin_indices[1,i]-1)*supercell_size+(bin_indices[2,i]-1)*supercell_size**2 for i in range(bin_indices.shape[1])])
            bincount = np.unique(atom_indices, return_counts=True)[1]
            if len(bincount) != supercell_size**3:
                raise TypeError("Incorrect number of bins. ")
            if max(bincount) != min(bincount):
                raise ValueError("Not all bins contain exactly the same number of atoms (1). ")
            
            B3denv = np.empty((3,supercell_size**2,supercell_size))
            for i in range(3):
                dims = [0,1,2]
                dims.remove(i)
                binx = bin_indices[dims,:]
                
                binned = [[] for _ in range(supercell_size**2)]
                for j in range(binx.shape[1]):
                    binned[(binx[0,j]-1)+supercell_size*(binx[1,j]-1)].append(j)
                binned = np.array(binned)

                
                for j in range(binned.shape[0]): # sort each bin in "i" direction coords
                    binned[j,:] = np.array([x for _, x in sorted(zip(cc[binned[j,:],i], binned[j,:]))]).reshape(1,supercell_size)
                
                B3denv[i,:] = binned
            
            B3denv = B3denv.astype(int)
            
            num_nn = math.ceil((supercell_size-1)/2)


            spatialnn = []
            spatialnorm = []
            for space in range(3):
                layernn = []
                layernorm = []
                for layer in range(supercell_size):
                    corrnn = np.empty((num_nn+1,3))
                    corrnorm = np.empty((num_nn+1,3))
                    for nn in range(num_nn+1):
                        pos1 = layer+(nn)
                        if pos1 > supercell_size-1:
                            pos1 -= (supercell_size)
                        pos2 = layer-(nn)
                        
                        tc1 = np.multiply(T[B3denv[space,:,layer],:],T[B3denv[space,:,pos1],:]).reshape(-1,3)
                        tc2 = np.multiply(T[B3denv[space,:,layer],:],T[B3denv[space,:,pos2],:]).reshape(-1,3)
                        tc = np.nanmean(np.concatenate((tc1,tc2),axis=0),axis=0)[np.newaxis,:]
                        tcnorm = (np.sqrt(np.abs(tc))*np.sign(tc))
                        #tc = np.cbrt(tc)
                        corrnn[nn,:] = tc
                        corrnorm[nn,:] = tcnorm
                    layernn.append(corrnn)
                    layernorm.append(corrnorm)
                layernn = np.nanmean(np.array(layernn),axis=0)
                layernn = layernn/layernn[0,:]
                spatialnn.append(layernn)
                layernorm = np.nanmean(np.array(layernorm),axis=0)
                layernorm = layernorm/layernorm[0,:]
                spatialnorm.append(layernorm)
                
            spatialnn = np.array(spatialnn)
            spatialnorm = np.array(spatialnorm)      
            
# =============================================================================
#             bin_indices = bin_indices-1 # 0-indexing
#             
#             num_nn = math.ceil((supercell_size-1)/2)
#             scmnorm = np.empty((bin_indices.shape[1],3,num_nn+1,3))
#             scmnorm[:] = np.nan
#             scm = np.empty((bin_indices.shape[1],3,num_nn+1,3))
#             scm[:] = np.nan
#             for o in range(bin_indices.shape[1]):
#                 si = bin_indices[:,[o]]
#                 for space in range(3):
#                     for n in range(num_nn+1):
#                         addit = np.array([[0],[0],[0]])
#                         addit[space,0] = n
#                         pos1 = si + addit
#                         pos1[pos1>supercell_size-1] = pos1[pos1>supercell_size-1]-supercell_size
#                         k1 = np.where(np.all(bin_indices==pos1,axis=0))[0][0]
#                         tc1 = np.multiply(T[o,:],T[k1,:])
#                         #tc1norm = np.sqrt(np.abs(tc1))*np.sign(tc1)
#                         #tc = np.nanmean(tc1,axis=0)
#                         #tcnorm = np.nanmean(tc1norm,axis=0)
#                         tcnorm = np.sqrt(np.abs(tc1))*np.sign(tc1)
#                         scmnorm[o,space,n,:] = tcnorm
#                         scm[o,space,n,:] = tc1
#                         
#             scm = scm/scm[:,:,[0],:]
#             scmnorm = scmnorm/scmnorm[:,:,[0],:]
#             spatialnn = np.mean(scm,axis=0)
#             spatialnorm = np.mean(scmnorm,axis=0)
# =============================================================================

            scdecay = quantify_tilt_domain(spatialnn,spatialnorm) # spatial decay length in 'unit cell'
            self.spatialCorrLength = scdecay 
            print(f"Tilting spatial correlation length: \n {np.round(scdecay,3)}")
            
            from pdyna.analysis import savitzky_golay, vis3D_domain_frame

            axisvis = 2
            ampfeat = T.copy()
            ampfeat[np.isnan(ampfeat)] = 0
            ampfeat[np.logical_or(ampfeat>23,ampfeat<-23)] = 0
            plotfeat = ampfeat[:,[axisvis]]
            
            #plotfeat = np.sqrt(np.abs(plotfeat))*np.sign(plotfeat)
            clipedges = (np.quantile(plotfeat,0.90)-np.quantile(plotfeat,0.10))/2
            print(f"The max tilting value of the supercell plot is +/- {round(clipedges,3)} degrees.")
            clipedges1 = [-3,3]
            plotfeat = np.clip(plotfeat,-clipedges,clipedges)
            if clipedges>4:
                plotfeat[np.logical_and(plotfeat<clipedges1[1],plotfeat>clipedges1[0])] = 0
            
            def map_rgb_tilt(x):
                #x = x/np.amax(np.abs(x))/2+0.5
                #return plt.cm.coolwarm(x)[:,:,0,:]
                x = x/np.amax(np.abs(x))
                p1 = np.array([0.196, 0.031, 0.318])
                p2 = np.array([0.918, 0.388, 0.161])
                pw = np.array([1, 1, 1])
                cx = np.empty((x.shape[0],3))
                cx[(x>=0)[:,0],:] = np.multiply(p1-pw,np.repeat(x[x>=0][:,np.newaxis],3,axis=1))+pw
                cx[(x<0)[:,0],:] = np.multiply(p2-pw,np.abs(np.repeat(x[x<0][:,np.newaxis],3,axis=1)))+pw
                return np.concatenate((cx,np.ones((cx.shape[0],1))),axis=1)
                
            
            cfeat = map_rgb_tilt(plotfeat)
            figname1 = f"Tilt3D_domain_frame_{uniname}"
            vis3D_domain_frame(cfeat,supercell_size,bin_indices,figname1)

        
        et1 = time.time()
        self.timing["tilt_distort"] = et1-et0
        

@dataclass
class Dataset:
    
    """
    Class representing the local configuration dataset to analyze.
    Initialize the class with reading the dataset.

    Parameters
    ----------
    data_format : data format
        Currently compatible format is 'mlab'.
    data_path : path of input files
        mlab: path of the ML_AB file
        
    """
    
    data_format: str = field(repr=False)
    data_path: str = field(repr=False)
    
    _Xsite_species = ['Cl','Br','I'] # update if your structrue contains other elements on the X-sites
    _Bsite_species = ['Pb','Sn'] # update if your structrue contains other elements on the B-sites
    
    # characteristic value of bond length of your material for structure construction 
    # the first interval covers the first and second NN of B-X (B-B) pairs, the second interval covers only the first NN of B-X (B-B) pairs.
    _fpg_val_BB = [[3,9.6], [6,8.8]] # empirical values for lead halide perovskites
    _fpg_val_BX = [[0.1,8], [3,6.8]] # empirical values for lead halide perovskites
    
    def __post_init__(self):
        
        et0 = time.time()

        if self.data_format == 'mlab':
            
            from pymlff import MLAB
            from pdyna.io import process_lat

            ab = MLAB.from_file(self.data_path)
            
            print("------------------------------------------------------------")
            print("Loading configuration files...")
            
            atomic_symbols = []
            clist = []
            lmlist = []
            llist = []
            
            for c in ab.configurations:
                ats = []
                for atype in c.atom_types:
                    ats.extend([atype]*c.atom_types_numbers[atype])
                atomic_symbols.append(ats)
                clist.append(c.coords)
                lmlist.append(c.lattice)
                llist.append(process_lat(c.lattice))
                    
            
            self.species = atomic_symbols
            self.Allpos = clist
            self.latmat = lmlist
            self.lattice = llist

            self.nframe = len(self.Allpos)

        else:
            raise TypeError("Unsupported data format: {}".format(self.data_format))
        
            
        et1 = time.time()
        self.timing = {}
        self.timing["reading"] = et1-et0
        
    
    def __str__(self):
        pattern = '''
        Perovskite Dataset
        Number of configurations: {}
        '''

        return pattern.format(self.nframe)
    
    
    def __repr__(self):
        return 'PDynA Dataset({} configurations)'.format(self.nframe)
    
    
    def featurize(self,
                 # general parameters
                 uniname = "test", # A unique user-defined name for this trajectory, will be used in printing and figure saving
                 saveFigures = False, # whether to save produced figures
                 tilt_corr_NN1 = True, # enable first NN correlation of tilting, reflecting the Glazer notation
                 ):
        
        from pdyna.structural import find_population_gap, apply_pbc_cart_vecs_single_frame
        from pdyna.structural import distance_matrix
        
        # pre-definitions
        print("Current dataset:",uniname)
        print("Configuration count:",self.nframe)

        # reset timing
        self.timing = {"reading": self.timing["reading"]}
        self.uniname = uniname
        
        print(" ")
        et0 = time.time()
        
        Tf = []
        Df = []
        if tilt_corr_NN1:
            Cf = []
        
        from tqdm import tqdm
        skipped = 0
        for f in tqdm(range(self.nframe)):
            
            atsyms = self.species[f]
            
            # read the coordinates and save   
            Xindex = []
            Bindex = []
            Cindex = []
            Nindex = []
            Hindex = []
            for i,site in enumerate(atsyms):
                 if site in self._Xsite_species:
                     Xindex.append(i)
                 if site in self._Bsite_species:
                     Bindex.append(i)  
                 if site == 'C':
                     Cindex.append(i)  
                 if site == 'N':
                     Nindex.append(i)  
                 if site == 'H':
                     Hindex.append(i)  
            
            Bpos = self.Allpos[f][Bindex,:]
            Xpos = self.Allpos[f][Xindex,:]
            
            mymat = self.latmat[f]
            
            r0=distance_matrix(Bpos,Bpos,mymat)
            search_NN1 = find_population_gap(r0, self._fpg_val_BB[0], self._fpg_val_BB[1])
            default_BB_dist = np.mean(r0[np.logical_and(r0>0.1,r0<search_NN1)])
            self.default_BB_dist = default_BB_dist
            
            Benv = []
            for B1, B2_list in enumerate(r0): # find the nearest Pb within a cutoff
                Benv.append([i for i,B2 in enumerate(B2_list) if (B2 > 0.1 and B2 < search_NN1)])
            Benv = np.array(Benv)
            
            try:
                aa = Benv.shape[1] # if some of the rows in Benv don't have 6 neighbours.
                if Benv.shape[1] != 3 and Benv.shape[1] != 6:
                    raise TypeError(f"The environment matrix is incorrect. The connectivity is {Benv.shape[1]} instead of 6. ")
                
            except IndexError:
                print(f"Need to adjust the range of B atom 1st NN distance (was {search_NN1}).  ")
                print("See the gap between the populations. \n")
                test_range = r0.reshape((1,r0.shape[0]**2))
                import matplotlib.pyplot as plt
                fig,ax = plt.subplots(1,1)
                plt.hist(test_range.reshape(-1,),range=[5.3,9.0],bins=100)
                #ax.scatter(test_range,test_range)
                #ax.set_xlim([5,10])
            
            self._Benv = Benv
            
            # label the constituent octahedra
            from pdyna.structural import fit_octahedral_network_defect_tol, octahedra_coords_into_bond_vectors, calc_distortions_from_bond_vectors
            from scipy.spatial.transform import Rotation as sstr
            from pdyna.structural import distance_matrix
            try: 
                rt = distance_matrix(Bpos,Xpos,mymat)
                neigh_list = fit_octahedral_network_defect_tol(Bpos,Xpos,rt,mymat,self._fpg_val_BX,1)
            except ValueError:
                skipped += 1
                continue
            except TypeError:
                skipped += 1
                continue

            disto = np.empty((0,4))
            Rmat = np.zeros((len(Bindex),3,3))
            Rmsd = np.zeros((len(Bindex),1))
            for B_site in range(len(Bindex)): # for each B-site atom
                raw = Xpos[neigh_list[B_site,:].astype(int),:] - Bpos[B_site,:]
                bx = octahedra_coords_into_bond_vectors(raw,mymat)
                dist_val,rotmat,rmsd = calc_distortions_from_bond_vectors(bx)
                Rmat[B_site,:] = rotmat
                Rmsd[B_site] = rmsd
                disto = np.concatenate((disto,dist_val.reshape(1,4)),axis = 0)
            
            T = np.zeros((len(Bindex),3))
            for i in range(Rmat.shape[0]):
                T[i,:] = sstr.from_matrix(Rmat[i,:]).as_euler('xyz', degrees=True)
            
            Tf.append(T)
            Df.append(disto)

            if tilt_corr_NN1:
                from pdyna.analysis import abs_sqrt
                Benv = self._Benv
                    
                if Benv.shape[1] == 3: # indicate a 2*2*2 supercell
                    
                    for i in range(Benv.shape[0]):
                        # for each Pb atom find its nearest Pb in each orthogonal direction. 
                        orders = np.argmax(np.abs(Bpos[Benv[i,:],:] - Bpos[i,:]), axis=0)
                        Benv[i,:] = Benv[i,:][orders] # correct the order of coords by 'x,y,z'
                    
                    # now each row of Benv contains the Pb atom index that sit in x,y and z direction of the row-numbered Pb atom.
                    Corr = np.empty((Benv.shape[0],3))
                    for B1 in range(Benv.shape[0]):
                            
                        Corr[B1,0] = abs_sqrt(T[B1,0]*T[Benv[B1,0],0])
                        Corr[B1,1] = abs_sqrt(T[B1,1]*T[Benv[B1,1],1])
                        Corr[B1,2] = abs_sqrt(T[B1,2]*T[Benv[B1,2],2])

                elif Benv.shape[1] == 6: # indicate a larger supercell
                    
                    Bcoordenv = self._Bcoordenv
                                    
                    ref_octa = np.array([[1,0,0],[-1,0,0],
                                         [0,1,0],[0,-1,0],
                                         [0,0,1],[0,0,-1]])

                    for i in range(Bcoordenv.shape[0]):
                        orders = np.zeros((1,6))
                        for j in range(6):
                            orders[0,j] = np.argmax(np.dot(Bcoordenv[i,:,:],ref_octa[j,:]))
                        Benv[i,:] = Benv[i,:][orders.astype(int)]
                            
                    # now each row of Benv contains the Pb atom index that sit in x,y and z direction of the row-numbered Pb atom.
                    Corr = np.empty((Benv.shape[0],6))
                    for B1 in range(Benv.shape[0]):
                            
                        Corr[B1,[0,1]] = abs_sqrt(T[[B1],0]*T[Benv[B1,[0,1]],0]) # x neighbour 1,2
                        Corr[B1,[2,3]] = abs_sqrt(T[[B1],1]*T[Benv[B1,[2,3]],1]) # y neighbour 1,2
                        Corr[B1,[4,5]] = abs_sqrt(T[[B1],2]*T[Benv[B1,[4,5]],2]) # z neighbour 1,2
                    
                else: 
                    raise TypeError(f"The environment matrix is incorrect. {Benv.shape[1]} ")
                
                Cf.append(Corr)
            
        if skipped != 0:
            print(f"Skipped {skipped} frames due to unrecognized structures.")
                
            
        self.Tilting = Tf
        self.Distortion = Df
        if tilt_corr_NN1:
            self.Tilting_Corr = Cf    
            
        
        if tilt_corr_NN1:
            Tm = np.empty((0,3))
            Dm = np.empty((0,4))
            Cm = np.empty((0,3))
            Fa = []
            for f in range(len(Tf)):
                Tm = np.concatenate((Tm,Tf[f]),axis=0)
                Dm = np.concatenate((Dm,Df[f]),axis=0)
                Cm = np.concatenate((Cm,Cf[f]),axis=0)
                Fa.append(np.concatenate((Df[f],Tf[f],Cf[f]),axis=1))
                
            Features = np.concatenate((Dm,Tm,Cm),axis=1)
        else:
            Tm = np.empty((0,3))
            Dm = np.empty((0,4))
            Fa = []
            for f in range(len(Tf)):
                Tm = np.concatenate((Tm,Tf[f]),axis=0)
                Dm = np.concatenate((Dm,Df[f]),axis=0)
                Fa.append(np.concatenate((Df[f],Tf[f]),axis=1))
                
            Features = np.concatenate((Dm,Tm),axis=1)
        
        self.FeatureVec = Features
        self.FeatureList = Fa
        
        # plotting
        from pdyna.analysis import draw_tilt_density, draw_tilt_and_corr_density_shade_longarray, draw_dist_density
        _,_ = draw_dist_density(Dm, self.uniname, saveFigures, n_bins = 100, title=None)
        if tilt_corr_NN1:
            _ = draw_tilt_and_corr_density_shade_longarray(Tm, Cm, self.uniname, saveFigures, title=self.uniname)
        else:
            draw_tilt_density(Tm, self.uniname, saveFigures, title=self.uniname)
        
            
        et1 = time.time()
        self.timing["tilt_distort"] = et1-et0
        self.timing["total"] = sum(list(self.timing.values()))
        print(" ")
        from pdyna.io import print_time
        print_time(self.timing)
        


