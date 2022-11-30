"""Core objects of PDyna."""

from __future__ import annotations

from dataclasses import dataclass, field
from pdyna.io import print_time
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
        Currently compatible formats are 'vasp' and 'lammps'.
    data_path : tuple of input files
        The input file path.
        vasp: (poscar_path, xdatcar_path, incar_path)
        lammps: (dump.out_path, MD setting tuple)
            MD setting tuple: (nsw,nblock,Ti,Tf,tstep)
    """
    
    data_format: str = field(repr=False)
    data_path: tuple = field(repr=False)
    
    
    def __post_init__(self):
        
        Xsite_species = ['Cl','Br','I'] # update if needed
        Bsite_species = ['Pb','Sn'] # update if needed
        
        et0 = time.time()

        if self.data_format == 'vasp':
            
            import pymatgen.io.vasp.inputs as vi
            import pymatgen.io.vasp.outputs as vo
            from pdyna.io import chemical_from_formula
            
            if len(self.data_path) != 3:
                raise TypeError("The input format for vasp must be (poscar_path, xdatcar_path, incar_path). ")
            poscar_path, xdatcar_path, incar_path = self.data_path
            
            print("------------------------------------------------------------")
            print("Loading Trajectory files...")
            
            # read POSCAR and XDATCAR files
            st0 = vi.Poscar.from_file(poscar_path,check_for_POTCAR=False).structure # initial configuration
            known_elem = ("I", "Br", "Cl", "Pb", "C", "H", "N", "Cs")
            for elem in st0.symbol_set:
                if not elem in known_elem:
                    raise ValueError(f"An unexpected element {elem} is found. ")
            self.st0 = st0
            self.natom = len(st0)
            self.species_set = st0.symbol_set
            self.formula = chemical_from_formula(st0)
            
            frames = vo.Xdatcar(xdatcar_path).structures
            st1 = frames[0]
            #self.st1 = st1
            self.nframe = len(frames)
            
            # check if initial structure has the same atom species order as the trajectory
            if not self.st0.atomic_numbers == st1.atomic_numbers:
                raise TypeError("The initial structure has a different atom species order with the trajectory")
            
            # read INCAR to obatin MD settings
            with open(incar_path,"r") as fp:
                lines = fp.readlines()
                nblock = 1
                Tf = None
                for line in lines:
                    if line.startswith('NBLOCK'):
                        nblock = int(line.split()[2])
                    if line.startswith('TEBEG'):
                        Ti = int(line.split()[2])
                        if Tf == None:
                            Tf = int(Ti)
                    if line.startswith('TEEND'):
                        Tf = int(line.split()[2])
                    if line.startswith('POTIM'):
                        tstep = float(line.split()[2])
                    if line.startswith('NSW'):
                        nsw = int(line.split()[2])
                self.MDsetting = {}
                self.MDsetting["nsw"] = nsw
                self.MDsetting["nblock"] = nblock
                self.MDsetting["Ti"] = Ti
                self.MDsetting["Tf"] = Tf
                self.MDsetting["tstep"] = tstep
                self.MDTimestep = tstep/1000*nblock  # the timestep between recorded frames
                self.Tgrad = (Tf-Ti)/(nsw*tstep/1000)   # temeperature gradient
        
        
        elif self.data_format == 'lammps':
            
            import pymatgen.io.ase as pia
            from pdyna.io import read_lammps_dump_text, chemical_from_formula
            
            if len(self.data_path) != 2:
                raise TypeError("The input format for lammps must be (dump.out_path, MD setting tuple). ")
            dump_path, lammps_setting = self.data_path    
            
            print("------------------------------------------------------------")
            print("Loading Trajectory files")
            
            frames = []
            with open(dump_path,"r") as fp:
                a=read_lammps_dump_text(fp)
                for i in a:
                    frames.append(pia.AseAtomsAdaptor.get_structure(i))

            st0 = frames[0]
            known_elem = ("I", "Br", "Cl", "Pb", "C", "H", "N", "Cs")
            for elem in st0.symbol_set:
                if not elem in known_elem:
                    raise ValueError(f"An unexpected element {elem} is found. ")
            del frames[0]
            
            self.st0 = st0
            self.natom = len(st0)
            st1 = frames[0]
            #self.st1 = st1
            self.species_set = st0.symbol_set
            self.formula = chemical_from_formula(st0)
            self.nframe = len(frames)
            
            self.MDsetting = {}
            self.MDsetting["nsw"] = lammps_setting[0]
            self.MDsetting["nblock"] = lammps_setting[1]
            self.MDsetting["Ti"] = lammps_setting[2]
            self.MDsetting["Tf"] = lammps_setting[3]
            self.MDsetting["tstep"] = lammps_setting[4]
            self.MDTimestep = lammps_setting[4]/1000*lammps_setting[1]  # the timestep between recorded frames
            self.Tgrad = (lammps_setting[3]-lammps_setting[2])/(lammps_setting[0]*lammps_setting[4]/1000)   # temeperature gradient
            
        else:
            raise TypeError("Unsupported data format: {}".format(self.data_format))
        
        
        # pre-definitions of the trajectory
        if 'C' in st0.symbol_set:
            self._flag_organic_A = True
        else:
            self._flag_organic_A = False
            
            
        # read the coordinates and save   
        Xindex = []
        Bindex = []
        Cindex = []
        Nindex = []
        Hindex = []
        for i,site in enumerate(st1.sites):
             if site.species_string in Xsite_species:
                 Xindex.append(i)
             if site.species_string in Bsite_species:
                 Bindex.append(i)  
             if site.species_string == 'C':
                 Cindex.append(i)  
             if site.species_string == 'N':
                 Nindex.append(i)  
             if site.species_string == 'H':
                 Hindex.append(i)  
                 
        Allpos = np.empty((self.nframe,self.natom,3))
        
        Allfrac = np.empty((self.nframe,self.natom,3))
        
        lattice = np.empty((self.nframe,6))
        latmat = np.empty((self.nframe,3,3))
        
        for fr,struct in enumerate(frames):

            Allpos[fr,:] = struct.cart_coords
            Allfrac[fr,:] = struct.frac_coords
            
            lattice[fr,:] = np.array([struct.lattice.abc,struct.lattice.angles]).reshape(1,6)
            latmat[fr,:] = struct.lattice.matrix
        
        self.Allpos = Allpos
        
        self.Bindex = Bindex
        self.Xindex = Xindex
        self.Cindex = Cindex
        self.Hindex = Hindex
        self.Nindex = Nindex
        
        self.Allfrac = Allfrac
        
        self.lattice = lattice
        self.latmat = latmat
            

    
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
                 uniname: str, # A unique user-defined name for this trajectory, will be used in printing and figure saving
                 read_mode: int, # key parameter, 1: equilibration mode, 2: quench/anneal mode
                 allow_equil = 0.5, # take the first x fraction of the trajectory as equilibration, this part will not be computed
                 read_every = 0, # read only every n steps, default is 0 which the code will decide an appropriate value according to the system size
                 saveFigures = False, # whether to save produced figures
                 lib_saver = False,  # whether to save computed material properties in lib file
                 lib_overwrite = False, # whether to overwrite existing lib entry, or just change upon them
                 
                 # function toggles
                 preset = 0, # 0: no preset, uses the toggles, 1: lat & tilt_distort, 2: lat & tilt_distort & tavg & MO, 3: all
                 toggle_lat = False, # switch of lattice parameter calculation
                 toggle_tavg = False, # switch of time averaged structure
                 toggle_tilt_distort = False, # switch of octahedral tilting and distortion calculation
                 toggle_MO = False, # switch of molecular orientation (MO) calculation (for organic A-site)
                 toggle_RDF = False, # switch of radial distribution function calculation
                 toggle_A_disp = False, # switch of A-site cation displacement calculation
                 
                 # Lattice parameter calculation
                 lat_method = 1, # lattice parameter analysis methods, 1: direct lattice cell dimension, 2: pseudo-cubic lattice parameter
                 zdir = 2, # specified z-direction in case of lat_method 2
                 lattice_rot = [0,0,0], # the rotation of system before lattice calculation in case of lat_method 2
                 smoother = True, # whether to use S-G smoothing on outputs 
                 leading_crop = 0.02, # remove the first x fraction of the trajectory on plotting 
                 
                 # time averaged structure
                 start_ratio = 0.5, # time-averaging structure ratio, e.g. 0.9 means only averaging the last 10% of trajectory
                 tavg_save_dir = ".\\", # directory for saving the time-averaging structures
                 
                 # octahedral tilting and distortion
                 multi_thread = 1, # if >1, enable multi-threading in this calculation, since not vectorized
                 orthogonal_frame = True, # only enable in 3C polytype to force orthogonal direction fiting                 
                 tilt_corr_NN1 = True, # enable first NN correlation of tilting, reflecting the Glazer notation
                 tilt_corr_spatial = False, # enable spatial correlation beyond NN1
                 octa_locality = False, # compute differentiated properties within mixed-halide sample
                 enable_refit = False, # refit the octahedral network in case of change of geometry
                 symm_8_fold = False, # tilting range, False: [-45,45], True: [0,45]
                 
                 # molecular orientation (MO)
                 MOautoCorr = False, # compute MO reorientation time constant
                 MO_corr_NN12 = False, # enable first and second NN correlation function of MO
                 ):
        
        # pre-definitions
        print("Current sample:",uniname)
        print(" ")
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
            toggle_A_disp = False
        elif preset == 2:
            toggle_lat = True
            toggle_tavg = True
            toggle_tilt_distort = True
            toggle_MO = True
            toggle_RDF = False
            toggle_A_disp = False
        elif preset == 3:
            toggle_lat = True
            toggle_tavg = True
            toggle_tilt_distort = True
            toggle_MO = True
            toggle_RDF = True
            toggle_A_disp = True
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
        
        angles = self.st0.lattice.angles
        sides = self.st0.lattice.abc
        if (max(angles) < 95 and min(angles) > 85) and (max(sides)-min(sides))/6 < 0.8:
            self._flag_cubic_cell = True
        else:
            self._flag_cubic_cell = False
        
        if read_mode == 2:
            allow_equil = 0
        
        if orthogonal_frame and self._flag_cubic_cell == False:
            print("!Warning: detected non-cubic cell but orthogonal octahedral reference is used! ")
        
        if self._flag_cubic_cell == False:
            lat_method = 1
            tilt_corr_NN1 = False
        
        if self.natom < 200:
            MO_corr_NN12 = False 
            tilt_corr_spatial = False 
            
        if read_every == 0:
            if self.natom < 400:
                read_every = 1
            else:
                if read_mode == 1:
                    read_every = round(self.nframe*(2.4e-05*(len(self.Bindex)**0.65))/(1-allow_equil))
                elif read_mode == 2:
                    read_every = round(self.nframe*(4.8e-05*(len(self.Bindex)**0.65))/(1-allow_equil))
                if read_every < 1:
                    read_every = 1
        
        self.Timestep = self.MDTimestep*read_every # timestep with read_every parameter skipping some steps regularly
        
        
        # label the constituent octahedra
        if toggle_tavg or toggle_tilt_distort: 
            from pdyna.structural import fit_octahedral_network
            
            st0Bpos = self.st0.cart_coords[self.Bindex,:]
            st0Xpos = self.st0.cart_coords[self.Xindex,:]
            mybox = np.array([self.st0.lattice.abc,self.st0.lattice.angles]).reshape(6,)
            mymat = self.st0.lattice.matrix
            
            if orthogonal_frame:
                neigh_list = fit_octahedral_network(st0Bpos,st0Xpos,mybox,mymat,orthogonal_frame)
                self.octahedra = neigh_list
            else:
                neigh_list, ref_initial = fit_octahedral_network(st0Bpos,st0Xpos,mybox,mymat,orthogonal_frame)
                self.octahedra = neigh_list
                self.octahedra_ref = ref_initial
        
        # label the constituent A-sites
        if toggle_MO or toggle_A_disp:
            from MDAnalysis.analysis.distances import distance_array
            
            st0 = self.st0
            
            Nindex = self.Nindex
            Cindex = self.Cindex
            
            Aindex_fa = []
            Aindex_ma = []
            
            mybox=np.array([st0.lattice.abc,st0.lattice.angles]).reshape(6,)
            dm = distance_array(st0.cart_coords[Cindex,:], st0.cart_coords[Nindex,:], mybox)   
            
            CN_max_distance = 2.5
            
            for i in range(dm.shape[0]):
                Ns = []
                temp = np.argwhere(dm[i,:] < CN_max_distance).reshape(-1)
                for j in temp:
                    Ns.append(Nindex[j])
                if len(temp) == 1:
                    Aindex_ma.append([Cindex[i],Ns])
                elif len(temp) == 2:
                    Aindex_fa.append([Cindex[i],Ns])
                else:
                    raise ValueError(f"There are {len(temp)} N atom connected to C atom number {i}")
                    
            Aindex_cs = []
            
            # search all A-site cations and their constituent atoms (if organic)
            for i,site in enumerate(st0.sites):
                 if site.species_string == 'Cs':
                     Aindex_cs.append(i)  
            
            self.A_sites = {"FA": Aindex_fa, "MA": Aindex_ma, "Cs": Aindex_cs }
        
        et1 = time.time()
        self.timing["env_resolve"] = et1-et0
        
        # use a lib file to store computed dynamic properties
        self.prop_lib = {}
        self.prop_lib['Ti'] = self.MDsetting['Ti']
        self.prop_lib['Tf'] = self.MDsetting['Tf']
        
        
        # running calculations
        if toggle_lat:
            self.lattice_parameter(lat_method=lat_method,uniname=uniname,read_mode=read_mode,allow_equil=allow_equil,zdir=zdir,lattice_rot=lattice_rot,smoother=smoother,leading_crop=leading_crop,saveFigures=saveFigures,title=title)
        
        if toggle_tavg:
            et0 = time.time()
            
            from pdyna.structural import structure_time_average_ase, simply_calc_distortion
            struct = structure_time_average_ase(self,start_ratio= start_ratio, cif_save_path=tavg_save_dir+f"\\{uniname}_tavg.cif")
            self.tavg_struct = struct
            tavg_dist = simply_calc_distortion(self)[0]
            print("time-averaged structure distortion mode: ")
            print(np.round(tavg_dist,4))
            print(" ")

            self.prop_lib['distortion_tavg'] = tavg_dist
            
            et1 = time.time()
            self.timing["tavg"] = et1-et0
        
        if toggle_tilt_distort:
            print("Computing octahedral tilting and distortion...")
            self.tilting_and_distortion(uniname=uniname,multi_thread=multi_thread,read_mode=read_mode,read_every=read_every,allow_equil=allow_equil,tilt_corr_NN1=tilt_corr_NN1,tilt_corr_spatial=tilt_corr_spatial,octa_locality=octa_locality,enable_refit=enable_refit, symm_8_fold=symm_8_fold,saveFigures=saveFigures,smoother=smoother,title=title,orthogonal_frame=orthogonal_frame)
            print("dynamic distortion:",np.round(self.prop_lib["distortion"][0],4))
            print("dynamic tilting:",np.round(self.prop_lib["tilting"].reshape(3,),3))
            if 'tilt_corr_polarity' in self.prop_lib:
                print("tilting correlation:",np.round(np.array(self.prop_lib['tilt_corr_polarity']).reshape(3,),3))
            print(" ")
            
        if toggle_MO:
            self.molecular_orientation(uniname=uniname,read_mode=read_mode,allow_equil=allow_equil,MOautoCorr=MOautoCorr, MO_corr_NN12=MO_corr_NN12, title=title,saveFigures=saveFigures,smoother=smoother)    
        
        if toggle_RDF:
            self.radial_distribution(allow_equil=allow_equil,uniname=uniname,saveFigures=saveFigures)
        
        if toggle_A_disp:
            self.A_site_displacement(allow_equil=allow_equil,uniname=uniname,saveFigures=saveFigures)
        
        if read_mode == 2 and MO_corr_NN12:
            from pdyna.analysis import draw_quench_properties
            draw_quench_properties(self.Lobj, self.Tobj, self.Mobj, uniname, saveFigures)
        
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
    
    
    
    def lattice_parameter(self,uniname,lat_method,read_mode,allow_equil,zdir,lattice_rot,smoother,leading_crop,saveFigures,title):
        
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
        lattice_rot: the rotation of system before lattice calculation in case of lat_method 2
        smoother: whether to use S-G smoothing on outputs 
        leading_crop: remove the first x fraction of the trajectory on plotting 
        """
        
        et0 = time.time()
        
        if lat_method == 1:
            Lat = np.empty((0,3))
            Lat[:] = np.NaN
            
            st0lat = self.st0.lattice
            std_param = [np.std(np.array([st0lat.a,st0lat.b,st0lat.c])),
                         np.std(np.array([st0lat.a/np.sqrt(2),st0lat.b/np.sqrt(2),st0lat.c/2]))]

            if std_param.index(min(std_param)) == 1:
                temp = self.lattice[round(self.nframe*allow_equil):,:3]
                temp[:,:2] = temp[:,:2]/np.sqrt(2)
                temp[:,2] = temp[:,2]/2
                Lat = temp
                    
            elif std_param.index(min(std_param)) == 0:
                Lat = self.lattice[round(self.nframe*allow_equil):,:3]
            
            if self._flag_cubic_cell: # if cubic cell
                Lat_scale = round(np.mean(Lat[0,:3])/6)
                Lat = Lat/Lat_scale
            else:
                print("!lattice_parameter: detected non-cubic geometry, use direct cell dimension as output. ")
            
            self.Lat = Lat
              
        elif lat_method == 2:
            
            from pdyna.structural import pseudocubic_lat
            Lat = pseudocubic_lat(self, allow_equil, zdrc=zdir, lattice_tilt=lattice_rot,orthor_filter=False)
            self.Lat = Lat
        
        else:
            raise TypeError("The lat_method parameter must be 1 or 2. ")
        
        num_crop = round(self.nframe*leading_crop)      
        
        
        # data visualization
        if read_mode == 1:
            if self.Tgrad == 0: # constant-T MD
                from pdyna.analysis import draw_lattice_density
                if self._flag_cubic_cell:
                    Lmu, Lstd = draw_lattice_density(Lat, uniname=uniname,saveFigures=saveFigures, n_bins = 50, num_crop = num_crop,screen = [5,8], title=title) 
                else:
                    Lmu, Lstd = draw_lattice_density(Lat, uniname=uniname,saveFigures=saveFigures, n_bins = 50, num_crop = num_crop, title=title) 
                    
            else: # changing-T MD
                if self.nframe*self.MDsetting["nblock"] < self.MDsetting["nsw"]*0.99: # with tolerance
                    print("Lattice: Incomplete run detected! \n")
                    Ti = self.MDsetting["Ti"]
                    Tf = self.MDsetting["Ti"]+(self.MDsetting["Tf"]-self.MDsetting["Ti"])*(self.nframe*self.MDsetting["nblock"]/self.MDsetting["nsw"])
                steps1 = np.linspace(Ti,Tf,self.nframe)

                invert_x = False
                if Tf<Ti:
                    invert_x = True
                
                from pdyna.analysis import draw_lattice_evolution
                draw_lattice_evolution(Lat, steps1, Tgrad = self.Tgrad, uniname=uniname, saveFigures = saveFigures, smoother = smoother, xaxis_type = 'T', Ti = Ti,invert_x=invert_x) 
            
            # update property lib
            self.prop_lib['lattice'] = [Lmu,Lstd]
            
        else: #quench mode
            timeline = np.linspace(1,Lat.shape[0],Lat.shape[0])*self.MDTimestep
            self.Ltimeline = timeline
            
            from pdyna.analysis import draw_lattice_evolution_time
            self.Lobj = draw_lattice_evolution_time(Lat, timeline, self.MDsetting["Ti"],uniname = uniname, saveFigures = saveFigures, smoother = smoother) 
            
        
        
        et1 = time.time()
        self.timing["lattice"] = et1-et0
        


    def tilting_and_distortion(self,uniname,multi_thread,read_mode,read_every,allow_equil,tilt_corr_NN1,tilt_corr_spatial,octa_locality,enable_refit, symm_8_fold,saveFigures,smoother,title,orthogonal_frame):
        
        """
        Octhedral tilting and distribution analysis.

        Parameters
        ----------
        multi_thread : number of multi-threading for this calculation, input 1 to disable
        orthogonal_frame : use True only for 3C polytype with octahedral coordination number of 6
        tilt_corr_NN1 : enable first NN correlation of tilting, reflecting the Glazer notation
        tilt_corr_spatial : enable spatial correlation beyond NN1
        octa_locality : compute differentiated properties within mixed-halide sample
            - configuration of each octahedron, giving 10 types according to halide geometry
            - quantify Br- and I-rich regions with concentration
        enable_refit : refit octahedral network when abnormal distortion values are detected (indicating change of network)
            - only turn on when rearrangement is observed
        symm_8_fold: enable to fold the negative axis of tilting status leaving angle in [0,45] degree

        """
        
        from MDAnalysis.analysis.distances import distance_array
        from pdyna.structural import resolve_octahedra, octahedra_coords_into_bond_vectors
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
        
        mybox=np.array([st0.lattice.abc,st0.lattice.angles]).reshape(6,)
        mymat=st0.lattice.matrix
        
        if octa_locality:
            from pdyna.structural import match_mixed_halide_octa
            
            b0 = st0.cart_coords[Bindex,:]
            x0 = st0.cart_coords[Xindex,:]
            
            Xspec = []
            Xonehot = np.empty((0,2))
            for i,site in enumerate([st0.sites[i] for i in Xindex]): 
                 if site.species_string == 'Br':
                     Xspec.append("Br")  
                     Xonehot = np.concatenate((Xonehot,np.array([[0,1]])),axis=0)
                 elif site.species_string == 'I':
                     Xspec.append("I")  
                     Xonehot = np.concatenate((Xonehot,np.array([[1,0]])),axis=0)
                 else:
                     raise TypeError(f"A X-site element {site.species_string} is found other than I and Br. ")
            
            r = distance_array(b0,x0,mybox)
            
            # compare with final config to make sure there is no change of env
            mybox1=self.lattice[-1,:]
            b1 = Bpos[-1,:]
            x1 = Xpos[-1,:]
            rf = distance_array(b1,x1,mybox1)
            if np.amax(r-rf) > 3: 
                raise ValueError(f"The difference between initial and final configs are too large ({np.amax(r-rf)})")
            
            octa_halide_code = [] # resolve the halides of a octahedron, key output
            octa_halide_code_single = []
            for B_site, X_list in enumerate(r): # for each B-site atom  
                
                raw = x0[neigh_list[B_site,:],:] - b0[B_site,:]
                bx = octahedra_coords_into_bond_vectors(raw,mymat)
                
                hals = []
                for j in range(6):
                    hals.append(Xspec[int(neigh_list[B_site,j])])
                
                form_factor, ff_single = match_mixed_halide_octa(bx,hals)
                octa_halide_code.append(form_factor) # determine each octa as one of the 10 configs, key output
                octa_halide_code_single.append(ff_single)
                
            # resolve local env of octahedron
            env_BX_distance = 10.7 # to cover approx. the third NN halides
            #plt.scatter(r.reshape(-1,),r.reshape(-1,))
            #plt.axvline(x=10.7)
            
            sampling = 50
            syscode = np.empty((0,len(Bindex),Xonehot.shape[1]))
            for fr in np.round(np.linspace(round(Bpos.shape[0]*allow_equil),Bpos.shape[0]-1,sampling)):
                fr = int(fr)
                
                mybox = lattice[fr,:]
                r = distance_array(Bpos[fr,:],Xpos[fr,:],mybox)
                Xcodemaster = np.empty((len(Bindex),Xonehot.shape[1]))
                for B_site, X_list in enumerate(r): # for each B-site atom
                    Xcoeff = [] # coefficient for proximity
                    Xcode = np.empty((0,Xonehot.shape[1])) # one-hot of halide identity
                    for xi, xval in enumerate(X_list):
                        if xval < env_BX_distance:
                            Xcode = np.concatenate((Xcode,Xonehot[xi,:].reshape(1,-1)),axis=0)
                            Xcoeff.append(1/(xval**1))
                    Xcoeff = (np.array(Xcoeff)/sum(Xcoeff)).reshape(1,-1)
                    Xcodemaster[B_site,:] = np.sum(np.multiply(np.transpose(Xcoeff),Xcode),axis=0)
                    
                syscode = np.concatenate((syscode,np.expand_dims(Xcodemaster,axis=0)),axis=0)
        
            syscode = np.average(syscode,axis=0) # label each B atom with its neighbouring halide density in one-hot manner, key output
        
        
        # tilting and distortion calculations
        ranger = self.nframe
        timeline = np.linspace(1,ranger,ranger)*self.MDTimestep
        if allow_equil == 0:
            ranger0 = 0
        elif allow_equil > 0:
            ranger0 = round(ranger*allow_equil)
            timeline = timeline[round(timeline.shape[0]*allow_equil):]
        
        readfr = []
        for fr in range(ranger0,ranger):
            if read_every != 1 and fr%read_every != 0:
                continue
            readfr.append(fr)
    
        if orthogonal_frame:
            ref_initial = None
        else:
            ref_initial = self.octahedra_ref 
        
        Di, T, refits = resolve_octahedra(Bpos,Xpos,readfr,enable_refit,multi_thread,lattice,latmat,neigh_list,orthogonal_frame,ref_initial)        
        
        if np.amax(Di) > 1:
            print(f"!Distortion: detected some distortion values ({np.amax(Di)}) larger than 1.")
            
        if np.amax(np.abs(T)) > 45:
            print(f"!Tilting: detected some tilting values ({np.amax(np.abs(T))}) outside the range -45 to 45 degree.")
        
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

        self.TDtimeline = timeline
        self.Distortion = Di
        self.Tilting = T
        
        
        # data visualization
        if read_mode == 2:
            from pdyna.analysis import draw_tilt_evolution_time
            self.Tobj = draw_tilt_evolution_time(T, timeline,uniname, saveFigures, smoother=smoother )

        
        elif self.Tgrad != 0: # read_mode 1 and changing-T MD
            from pdyna.analysis import draw_distortion_evolution_sca, draw_tilt_evolution_sca
        
            steps = np.linspace(self.MDsetting["Ti"],self.MDsetting["Tf"],Bpos.shape[0])
            if read_every != 0:
                temp_list = []
                for i,temp in enumerate(steps):
                    if i%read_every == 0:
                        temp_list.append(temp)
                steps = np.array(temp_list)
                
                assert steps.shape[0] == Di.shape[0]
                assert steps.shape[0] == T.shape[0]
            
            draw_distortion_evolution_sca(Di, steps, uniname, saveFigures, xaxis_type = 'T', scasize = 1)
            draw_tilt_evolution_sca(T, steps, uniname, saveFigures, xaxis_type = 'T', scasize = 1)

        else: # read_mode 1, constant-T MD (equilibration)
            from pdyna.analysis import draw_dist_density, draw_tilt_density
            Dmu,Dstd = draw_dist_density(Di, uniname, saveFigures, n_bins = 100, title=None)
            if not tilt_corr_NN1:
                draw_tilt_density(T, uniname, saveFigures,symm_8_fold=symm_8_fold,title=title)
                
        
        if octa_locality:
            from pdyna.analysis import draw_octatype_tilt_density, draw_octatype_dist_density
            from pdyna.analysis import draw_halideconc_tilt_density, draw_halideconc_dist_density
            
            # categorize the octa into different configs
            config_types = set(octa_halide_code_single)
            typelib = [[] for numm in range(len(config_types))]
            for ti, typei in enumerate(config_types):
                typelib[ti] = [k for k, x in enumerate(octa_halide_code_single) if x == typei]
                
            Dtype = []
            Ttype = []
            for ti, types in enumerate(typelib):
                Dtype.append(Di[:,types,:])
                Ttype.append(T[:,types,:])
            
            Tmaxs_type = draw_octatype_tilt_density(Ttype, config_types, uniname, saveFigures)
            Dgauss_type, Dgaussstd_type = draw_octatype_dist_density(Dtype, config_types, uniname, saveFigures)

            
            # activate local Br content analysis only if the sample is large enough
            brrange = [np.amin(syscode[:,1]),np.amax(syscode[:,1])]
            brbinnum = 10
            diffbin = (brrange[1]-brrange[0])/brbinnum*0.5
            binrange = [np.amin(syscode[:,1])-diffbin,np.amax(syscode[:,1])+diffbin]
            if syscode.shape[0] >= 64 and brrange[1]-brrange[0] > 0.2: 
                Bins = np.linspace(binrange[0],binrange[1],brbinnum+1)
                bininds = np.digitize(syscode[:,1],Bins)-1
                brbins = [[] for kk in range(brbinnum)] # global index of B atoms that are in each bin of Bins
                for ibin, binnum in enumerate(bininds):
                    brbins[binnum].append(ibin)
                    
            bincent = (Bins[1:]+Bins[:-1])/2
            
            concent = [] # concentrations recorded
            Dconc = []
            Tconc = []
            for ii,item in enumerate(brbins):
                if len(item) == 0: continue
                concent.append(bincent[ii])
                Dconc.append(Di[:,item,:])
                Tconc.append(T[:,item,:])
            
            Tmaxs_conc = draw_halideconc_tilt_density(Tconc, concent, uniname, saveFigures)
            Dgauss_conc, Dgaussstd_conc = draw_halideconc_dist_density(Dconc, concent, uniname, saveFigures)
        
        if read_mode == 1:
            Tval = np.array(compute_tilt_density(T)).reshape((3,1))
            self.prop_lib['distortion'] = [Dmu,Dstd]
            self.prop_lib['tilting'] = Tval
        
        
        # NN1 correlation function of tilting (Glazer notation)
        if tilt_corr_NN1:
            from pdyna.analysis import abs_sqrt, draw_tilt_corr_evolution_sca, draw_tilt_and_corr_density_shade
            
            default_BB_dist = 6.2
            
            ri=distance_array(Bpos[0,:],Bpos[0,:],self.lattice[0,:])
            rf=distance_array(Bpos[-1,:],Bpos[-1,:],self.lattice[-1,:])

            if np.amax(np.abs(ri-rf)) > 3: # confirm that no change in the Pb framework
                print("!Tilt-spatial: The difference between the initial and final distance matrix is above threshold ({}), check ri and rf. \n".format(round(np.amax(np.abs(ri-rf)),3)))
            
            search_NN1 = 7.1
            
            Benv = []
            for B1, B2_list in enumerate(ri): # find the nearest Pb within a cutoff
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
                
                
            if Benv.shape[1] == 3: # indicate a 2*2*2 supercell
                
                #ref_coords = np.array([[0,0,0],[-6.15,0,0],[0,-6.15,0],[0,0,-6.15]])
                
                for i in range(Benv.shape[0]):
                    # for each Pb atom find its nearest Pb in each orthogonal direction. 
                    orders = np.argmax(np.abs(Bpos[0,Benv[i,:],:] - Bpos[0,i,:]), axis=0)
                    Benv[i,:] = Benv[i,:][orders] # correct the order of coords by 'x,y,z'
                
                # now each row of Benv contains the Pb atom index that sit in x,y and z direction of the row-numbered Pb atom.
                Corr = np.empty((T.shape[0],T.shape[1],3))
                for fr in range(T.shape[0]):
                    for B1 in range(T.shape[1]):
                        
                        Corr[fr,B1,0] = abs_sqrt(T[fr,B1,0]*T[fr,Benv[B1,0],0])
                        Corr[fr,B1,1] = abs_sqrt(T[fr,B1,1]*T[fr,Benv[B1,1],1])
                        Corr[fr,B1,2] = abs_sqrt(T[fr,B1,2]*T[fr,Benv[B1,2],2])
                
                # Normalize Corr
                #Corr = Corr/np.amax(Corr)

            elif Benv.shape[1] == 6: # indicate a larger supercell
                
                lbb = np.average(st0.lattice.abc)
                
                Bcoordenv = np.empty((Benv.shape[0],6,3))
                for i in range(Benv.shape[0]):
                    Bcoordenv[i,:] = Bpos[0,Benv[i,:],:] - Bpos[0,i,:]
                    
                for ix in range(Bcoordenv.shape[0]):
                    for iy in range(Bcoordenv.shape[1]):
                        for iz in range(Bcoordenv.shape[2]):
                            if abs(Bcoordenv[ix,iy,iz]) > default_BB_dist*1.5:
                                if Bcoordenv[ix,iy,iz] > 0:
                                    Bcoordenv[ix,iy,iz] = Bcoordenv[ix,iy,iz]-lbb
                                else:
                                    Bcoordenv[ix,iy,iz] = Bcoordenv[ix,iy,iz]+lbb
                                
                ref_octa = np.array([[1,0,0],[-1,0,0],
                                     [0,1,0],[0,-1,0],
                                     [0,0,1],[0,0,-1]])
                for i in range(Bcoordenv.shape[0]):
                    orders = np.zeros((1,6))
                    for j in range(Bcoordenv.shape[1]):
                        orders[0,j] = np.argmax(np.dot(ref_octa,Bcoordenv[i,j,:]))
                    Benv[i,:] = Benv[i,:][orders.astype(int)]
                        
                
                # now each row of Benv contains the Pb atom index that sit in x,y and z direction of the row-numbered Pb atom.
                Corr = np.empty((T.shape[0],T.shape[1],6))
                for fr in range(T.shape[0]):
                    for B1 in range(T.shape[1]):
                        
                        Corr[fr,B1,0] = abs_sqrt(T[fr,B1,0]*T[fr,Benv[B1,0],0])
                        Corr[fr,B1,1] = abs_sqrt(T[fr,B1,0]*T[fr,Benv[B1,1],0])
                        Corr[fr,B1,2] = abs_sqrt(T[fr,B1,1]*T[fr,Benv[B1,2],1])
                        Corr[fr,B1,3] = abs_sqrt(T[fr,B1,1]*T[fr,Benv[B1,3],1])
                        Corr[fr,B1,4] = abs_sqrt(T[fr,B1,2]*T[fr,Benv[B1,4],2])
                        Corr[fr,B1,5] = abs_sqrt(T[fr,B1,2]*T[fr,Benv[B1,5],2])
                
                # Normalize Corr
                # Corr = Corr/np.amax(Corr)
                
            else: 
                raise TypeError(f"The environment matrix is incorrect. {Benv.shape[1]} ")
                
            self.Tilting_Corr = Corr
            
            if self.Tgrad != 0:
                draw_tilt_corr_evolution_sca(Corr, steps, uniname, saveFigures, xaxis_type = 'T') 
            else:
                polarity = draw_tilt_and_corr_density_shade(T,Corr, uniname, saveFigures,title=title)
                self.prop_lib["tilt_corr_polarity"] = polarity

        
        if tilt_corr_spatial:
            import math
            from scipy.stats import binned_statistic_dd as binstat
            from pdyna.analysis import draw_tilt_spacial_corr

            cc = st0.cart_coords[Bindex,:]
            cell_lat = st0.lattice.abc
            
            supercell_size = round(np.mean(cell_lat)/default_BB_dist)
            
            bin_indices = binstat(cc, None, 'count', bins=[supercell_size,supercell_size,supercell_size], 
                                  range=[[np.amin(cc[:,0])-0.5*cell_lat[0]/supercell_size, 
                                          np.amax(cc[:,0])+0.5*cell_lat[0]/supercell_size], 
                                         [np.amin(cc[:,1])-0.5*cell_lat[1]/supercell_size, 
                                          np.amax(cc[:,1])+0.5*cell_lat[1]/supercell_size],
                                         [np.amin(cc[:,2])-0.5*cell_lat[2]/supercell_size, 
                                          np.amax(cc[:,2])+0.5*cell_lat[2]/supercell_size]],
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

            tt = [[[],[],[]] for _ in range(num_nn)]
            for fr in range(T.shape[0]):
                for i in range(supercell_size**2):
                    for at in range(supercell_size):
                        for dire in range(3):    
                            for nn in range(num_nn):
                                pos1 = at+(nn+1)
                                if pos1 > supercell_size-1:
                                    pos1 -= supercell_size
                                pos2 = at-(nn+1)
                                
                                temp = T[fr,B3denv[dire,i,at],dire]*T[fr,B3denv[dire,i,pos1],dire]
                                temp = np.sqrt(np.abs(temp))*np.sign(temp)
                                tt[nn][dire].append(temp)
                                temp = T[fr,B3denv[dire,i,at],dire]*T[fr,B3denv[dire,i,pos2],dire]
                                temp = np.sqrt(np.abs(temp))*np.sign(temp)
                                tt[nn][dire].append(temp)
                        
            lenCorr=np.array(tt)
            self.lenCorr = lenCorr
            
            draw_tilt_spacial_corr(lenCorr, uniname, saveFigures, n_bins = 100)

        et1 = time.time()
        self.timing["tilt_distort"] = et1-et0
        
    
    def molecular_orientation(self,uniname,read_mode,allow_equil,MOautoCorr,MO_corr_NN12,title,saveFigures,smoother):
        """
        A-site molecular orientation (MO) analysis.

        Parameters
        ----------
        MOautoCorr : calculate MO decorrelation time constant
        MO_corr_NN12 : compute first and second NN MO correlation functions

        """
        
        et0 = time.time()
        
        from MDAnalysis.analysis.distances import distance_array
        from pdyna.analysis import MO_correlation, orientation_density, orientation_density_2pan, fit_exp_decay, orientation_density_3D
        from pdyna.structural import apply_pbc_cart_vecs
        
        Afa = self.A_sites["FA"]
        Ama = self.A_sites["MA"]
        
        lattice = self.lattice
        Cpos = self.Allpos[:,self.Cindex,:]
        Npos = self.Allpos[:,self.Nindex,:]
        
        trajnum = list(range(round(self.nframe*allow_equil),self.nframe))
        latmat = self.latmat[trajnum,:]
        
        CNdiff = np.amax(np.abs(distance_array(Cpos[0,:],Npos[0,:],lattice[0,:])-distance_array(Cpos[-1,:],Npos[-1,:],lattice[-1,:])))
        
        if CNdiff > 5:
            print(f"!MO: A change of C-N connectivity is detected (ref value {CNdiff}).")

        if len(Ama) > 0:
            Clist = [i[0] for i in Ama]
            Nlist = [i[1][0] for i in Ama]
            
            Cpos = self.Allpos[trajnum,:][:,Clist,:]
            Npos = self.Allpos[trajnum,:][:,Nlist,:]
            
            cn = Cpos-Npos
            cn = apply_pbc_cart_vecs(cn,latmat)
            CN = np.divide(cn,np.expand_dims(np.linalg.norm(cn,axis=2),axis=2))
            
            self.MA_MOvec = CN
            
            orientation_density(CN,saveFigures,uniname,title=title)
            #orientation_density_3D(CN,"MA",saveFigures,uniname)
            
            
        if len(Afa) > 0:
            
            Clist = [i[0] for i in Afa]
            N1list = [i[1][0] for i in Afa]
            N2list = [i[1][1] for i in Afa]
            
            Nlist = N1list+N2list
            
            Cpos = self.Allpos[trajnum,:][:,Clist,:]
            N1pos = self.Allpos[trajnum,:][:,N1list,:]
            N2pos = self.Allpos[trajnum,:][:,N2list,:]
            
            cn1 = Cpos-N1pos
            cn2 = Cpos-N2pos
            CN1 = apply_pbc_cart_vecs(cn1,latmat)
            CN2 = apply_pbc_cart_vecs(cn2,latmat)
            
            CN = CN1+CN2
            CN = np.divide(CN,np.expand_dims(np.linalg.norm(CN,axis=2),axis=2))
            
            nn = N1pos-N2pos
            NN = apply_pbc_cart_vecs(nn,latmat)
            NN = np.divide(NN,np.expand_dims(np.linalg.norm(NN,axis=2),axis=2))
            
            self.FA_MOvec1 = CN
            self.FA_MOvec2 = NN
            
            orientation_density_2pan(CN,NN,saveFigures,uniname,title=title)
            #orientation_density_3D(CN,"FA1",saveFigures,uniname)
            #orientation_density_3D(NN,"FA2",saveFigures,uniname)
            

        if MOautoCorr is True:
            if len(Afa) > 0 and len(Ama) > 0:
                raise TypeError("Need to write code for both species here")
            #sys.stdout.flush()
            corrtime, autocorr = MO_correlation(CN,self.MDTimestep,False,uniname)
            self.MO_autocorr = np.concatenate((corrtime,autocorr),axis=0)
            tconst = fit_exp_decay(corrtime, autocorr)

            print("MO decorrelation time: "+str(round(tconst,4))+' ps')
            if tconst < 0:
                print("!MO: Negative decorrelation time constant is found, please check if the trajectory is too short or system size too small. ")
            print(" ")
            
            self.prop_lib['reorientation'] = tconst
       
        
        if MO_corr_NN12 and not (len(Afa) > 0 and len(Ama) > 0):
            import math
            from pdyna.analysis import draw_MO_spacial_corr_time, draw_MO_spacial_corr_NN12, draw_MO_spacial_corr
            
            st0 = self.st0
            cc = st0.cart_coords[Clist,:]
            cfc = st0.frac_coords[Clist,:]
            nfc = st0.frac_coords[Nlist,:]
            
            dm = st0.distance_matrix[Clist][:,Nlist]

            cell_lat = st0.lattice.abc
            
            supercell_size = round(len(self.Bindex)**(1/3))
            
            if np.linalg.norm([np.amin(cc[:,0]),np.amin(cc[:,1]),np.amin(cc[:,2])]) < 1: # the first carbon atom is too close to the origin which will lead to reading error
                cc = cc+3 # shift the structure
                for i in range(cc.shape[0]):
                    for j in range(cc.shape[1]):
                        if cc[i,j] > cell_lat[j]:
                            cc[i,j] = cc[i,j]-cell_lat[j]
            
            
            CNbondmax = 2.2
            mol_centers = np.empty((0,3))
            for i in range(cc.shape[0]):
                dists = dm[i,:]
                bondedN = np.where(dists<CNbondmax)
                Nfs = np.squeeze(nfc[bondedN,:])
                if Nfs.ndim == 1:
                    Nfs = Nfs.reshape(1,3)
                Ns = Nfs
                for j in range(Ns.shape[0]):
                    jimage = st0.lattice.get_distance_and_image(cfc[i,:],Nfs[j,:])[1]
                    Ns[j,:] = st0.lattice.get_cartesian_coords(jimage + Nfs[j,:])
                
                centi = np.mean(np.concatenate((Ns,cc[[i],:]),axis=0),axis=0)
                mol_centers = np.concatenate((mol_centers,centi.reshape(1,3)),axis=0)
            
            blist = []
            for i,site in enumerate(st0.sites):
                if site.species_string in ['Pb', 'Sn']:
                    blist.append(i)
            celldim = np.cbrt(len(blist))
            bdm = st0.distance_matrix[blist][:,blist]
            bdm = bdm.reshape((bdm.shape[0]**2,1))
            BBdist = np.mean(bdm[np.logical_and(bdm>0.1,bdm<7)]) # Pb-Pb distance as molecule separations
            
            mol_min = np.min(mol_centers,axis=0)
            
            grided = np.round((mol_centers-mol_min)/BBdist)
            for i in range(grided.shape[0]):
                for j in range(grided.shape[1]):
                    if grided[i,j] == celldim:
                        grided[i,j] = grided[i,j]-celldim
            counts = np.unique(grided, return_counts=True)[1]
            if max(counts) != 3*celldim**2 or min(counts) != 3*celldim**2:
                raise TypeError("The molecules did not fit in the grids. ")
            
            bin_indices = (np.transpose(grided)+1).astype(int) # match with the below format        

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
            for fr in range(CN.shape[0]):
                for i in range(supercell_size**2):
                    for at in range(supercell_size):
                        for dire in range(3):    
                            for nn in range(num_nn):
                                pos1 = at+(nn+1)
                                if pos1 > supercell_size-1:
                                    pos1 -= supercell_size
                                #pos2 = at-(nn+1)
                                
                                temp = np.dot(CN[fr,C3denv[dire,i,at],:],CN[fr,C3denv[dire,i,pos1],:])
                                arr[nn,dire,fr,i*supercell_size+at] = temp
                                #temp = np.dot(cnsn[fr,C3denv[dire,i,at],:],cnsn[fr,C3denv[dire,i,pos2],:])
                                #tt[nn][dire].append(temp)
            if np.isnan(np.sum(arr)): raise TypeError("Some element missing in array.")
            
            MOCorr = arr
            self.MOCorr = MOCorr
            
            if read_mode == 2:
                Mobj = draw_MO_spacial_corr_time(MOCorr, self.Ltimeline, uniname, saveFigures, smoother=smoother)
                self.Mobj = Mobj
                
            elif read_mode == 1:
                draw_MO_spacial_corr_NN12(MOCorr, uniname, saveFigures) 
                draw_MO_spacial_corr(MOCorr, uniname, saveFigures = False)

        et1 = time.time()
        self.timing["MO"] = et1-et0
    
    
    def radial_distribution(self,allow_equil,uniname,saveFigures):
        """
        Radial distribution function (RDF) analysis.

        Parameters
        ----------

        """
        
        et0 = time.time()
        
        from MDAnalysis.analysis.distances import distance_array
        from pdyna.analysis import draw_RDF
        
        trajnum = list(range(round(self.nframe*allow_equil),self.nframe))
        
        Xindex = self.Xindex
        Bindex = self.Bindex
        Cpos = self.Allpos[:,self.Cindex,:]
        Npos = self.Allpos[:,self.Nindex,:]
        Bpos = self.Allpos[:,Bindex,:]
        Xpos = self.Allpos[:,Xindex,:]
        
        CNtol = 2.7
        BXtol = 12.8
        
        if self._flag_organic_A:
            
            assert len(Xindex)/len(Bindex) == 3
            
            #if (len(Nindex)/len(Cindex)).is_integer():
            #    NCratio = int(len(Nindex)/len(Cindex))
            #else:
            #    raise TypeError("The stiochiometric ratio between C and N is not an integer. ")
            

            CNda = np.empty((0,))
            BXda = np.empty((0,))
            for i,fr in enumerate(trajnum):
                mybox=self.lattice[fr,:]
                #if i == 0:
                #    CNr1=distance_array(Cpos[fr,:],Npos[fr,:],mybox)
                #    BXr1=distance_array(Bpos[fr,:],Xpos[fr,:],mybox)
                #if i == len(trajnum)-1:
                #    CNr2=distance_array(Cpos[fr,:],Npos[fr,:],mybox)
                #    BXr2=distance_array(Bpos[fr,:],Xpos[fr,:],mybox)
                    
                CNr=distance_array(Cpos[fr,:],Npos[fr,:],mybox)
                BXr=distance_array(Bpos[fr,:],Xpos[fr,:],mybox)
                
                #assert (CNr<CNtol).sum() == Natoms.n_atoms
                #assert (BXr<BXtol).sum() == atoms_Bsite.n_atoms*6
                
                CNda = np.concatenate((CNda,CNr[CNr<CNtol]),axis = 0)
                BXda = np.concatenate((BXda,BXr[BXr<BXtol]),axis = 0)
                
            #assert np.amax(CNr1-CNr2)<4
            #assert np.amax(BXr1-BXr2)<2.5
            
            #if saveFigures:
            #    af.draw_RDF(CNda, title = 'C-N RDF', fig_name = f"{uniname}_CN_RDF.png")
            #    af.draw_RDF(BXda, title = 'B-X RDF', fig_name = f"{uniname}_BX_RDF.png")
            #else:
            draw_RDF(BXda, "BX", uniname, False)
            draw_RDF(CNda, "CN", uniname, False)
            ccn,bcn1 = np.histogram(CNda,bins=100,range=[1.38,1.65])
            bcn = 0.5*(bcn1[1:]+bcn1[:-1])
            cbx,bbx1 = np.histogram(BXda,bins=300,range=[0,12])
            bbx = 0.5*(bbx1[1:]+bbx1[:-1])
            BXRDF = bbx,cbx
            CNRDF = bcn,ccn
            
            self.BX_RDF = BXRDF
            self.CN_RDF = CNRDF
        
        else:
            
            assert len(Xindex)/len(Bindex) == 3
            
            #if (len(Nindex)/len(Cindex)).is_integer():
            #    NCratio = int(len(Nindex)/len(Cindex))
            #else:
            #    raise TypeError("The stiochiometric ratio between C and N is not an integer. ")
            

            BXda = np.empty((0,))
            for i,fr in enumerate(trajnum):
                mybox=self.lattice[fr,:]
                if i == 0:
                    BXr1=distance_array(Bpos[fr,:],Xpos[fr,:],mybox)
                if i == len(trajnum)-1:
                    BXr2=distance_array(Bpos[fr,:],Xpos[fr,:],mybox)
                    
                BXr=distance_array(Bpos[fr,:],Xpos[fr,:],mybox)
                
                #assert (BXr<BXtol).sum() == atoms_Bsite.n_atoms*6
                
                BXda = np.concatenate((BXda,BXr[BXr<BXtol]),axis = 0)
                
            #assert np.amax(BXr1-BXr2)<2.5
            
            draw_RDF(BXda, "BX", uniname, False)
            cbx,bbx1 = np.histogram(BXda,bins=300,range=[0,12])
            bbx = 0.5*(bbx1[1:]+bbx1[:-1])
            BXRDF = bbx,cbx
            
            self.BX_RDF = BXRDF
        
        et1 = time.time()
        self.timing["RDF"] = et1-et0


    def A_site_displacement(self,allow_equil,uniname,saveFigures):
        
        """
        A-site cation displacement analysis.

        Parameters
        ----------

        """

        from MDAnalysis.analysis.distances import distance_array
        from pdyna.structural import centmass_organic, centmass_organic_vec, find_B_cage_and_disp
        from pdyna.analysis import fit_3D_disp_atomwise, fit_3D_disp_total, peaks_3D_scatter
        
        et0 = time.time()
        
        st0 = self.st0
        st0pos = self.st0.cart_coords
        Allpos = self.Allpos
        lattice = self.lattice
        latmat = self.latmat
        
        Bindex = self.Bindex
        Hindex = self.Hindex
        
        Afa = self.A_sites["FA"]
        Ama = self.A_sites["MA"]
        Aindex_cs = self.A_sites["Cs"]
        
        ABsep = 8.2
        mybox=np.array([st0.lattice.abc,st0.lattice.angles]).reshape(6,)
        st0Bpos = st0pos[Bindex,:]
        
        CN_H_tol = 1.35
        
        Aindex_fa = []
        Aindex_ma = []
        
        B8envs = {}
        
        if len(Afa) > 0:
            for i,env in enumerate(Afa):
                dm = distance_array(st0pos[[env[0]]+env[1],:], st0pos[Hindex,:],mybox)
                Hs = sorted(list(np.argwhere(dm<CN_H_tol)[:,1]))
                Aindex_fa.append(env+[Hs])
                
                cent = centmass_organic(st0pos,st0.lattice.matrix,env+[Hs])
                ri=distance_array(cent,st0Bpos,mybox) 
                Bs = []
                for j in range(ri.shape[1]):
                    if ri[0,j] < ABsep:
                        Bs.append(Bindex[j])
                try:
                    assert len(Bs) == 8
                    B8envs[env[0]] = Bs
                except AssertionError: # can't find with threshold distance, try using nearest 8 atoms
                    cent = centmass_organic_vec(Allpos,latmat,env+[Hs])
                    ri = np.empty((Allpos.shape[0],len(Bindex)))
                    for fr in range(Allpos.shape[0]):
                        ri[fr,:,]=distance_array(cent[fr,:],Allpos[fr,Bindex,:],lattice[fr,:]) 
                    ri = np.expand_dims(np.average(ri,axis=0),axis=0)
                    
                    Bs = []
                    for j in range(ri.shape[1]):
                        if ri[0,j] < ABsep:
                            Bs.append(Bindex[j])
                    assert len(Bs) == 8
                    B8envs[env[0]] = Bs
        
        if len(Ama) > 0:
            for i,env in enumerate(Ama):
                dm = distance_array(st0pos[[env[0]]+env[1],:], st0pos[Hindex,:],mybox)
                Hs = sorted(list(np.argwhere(dm<CN_H_tol)[:,1]))
                Aindex_ma.append(env+[Hs])
                
                cent = centmass_organic(st0pos,st0.lattice.matrix,env+[Hs])
                ri=distance_array(cent,st0Bpos,mybox) 
                
                Bs = []
                for j in range(ri.shape[1]):
                    if ri[0,j] < ABsep:
                        Bs.append(Bindex[j])
                try:
                    assert len(Bs) == 8
                    B8envs[env[0]] = Bs
                except AssertionError: # can't find with threshold distance, try using nearest 8 atoms
                    cent = centmass_organic_vec(Allpos,latmat,env+[Hs])
                    ri = np.empty((Allpos.shape[0],len(Bindex)))
                    for fr in range(Allpos.shape[0]):
                        ri[fr,:,]=distance_array(cent[fr,:],Allpos[fr,Bindex,:],lattice[fr,:]) 
                    ri = np.expand_dims(np.average(ri,axis=0),axis=0)
                    
                    Bs = []
                    for j in range(ri.shape[1]):
                        if ri[0,j] < ABsep:
                            Bs.append(Bindex[j])
                    assert len(Bs) == 8
                    B8envs[env[0]] = Bs
                
                
        if len(Aindex_cs) > 0:
            for i,env in enumerate(Aindex_cs):

                ri=distance_array(st0.cart_coords[env,:],st0Bpos,mybox) 
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
                disp_ma[:,ai,:] = find_B_cage_and_disp(Allposfr,latmatfr,cent,B8envs[envs[0]])
            
            self.disp_ma = disp_ma
            dispvec_ma = disp_ma.reshape(-1,3)
            
            moltype = "MA"
            peaks_ma = fit_3D_disp_atomwise(disp_ma,readTimestep,uniname,moltype,saveFigures,title=moltype)
            fit_3D_disp_total(dispvec_ma,uniname,moltype,saveFigures,title=moltype)
            peaks_3D_scatter(peaks_ma,uniname,moltype,saveFigures)
            
        if len(Aindex_fa) > 0:
            disp_fa = np.empty((Allposfr.shape[0],len(Aindex_fa),3))
            for ai, envs in enumerate(Aindex_fa):
                cent = centmass_organic_vec(Allposfr,latmatfr,envs)
                disp_fa[:,ai,:] = find_B_cage_and_disp(Allposfr,latmatfr,cent,B8envs[envs[0]])
            
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
        
        
        et1 = time.time()
        self.timing["A_disp"] = et1-et0

