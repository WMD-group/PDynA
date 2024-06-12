# choose an initialization with respect to your trajectory type
# VASP XDATCAR
traj = Trajectory("vasp",(poscar_path, xdatcar_path, incar_path))

# or LAMMPS dump file
MDtup = (100,100,1) # tuple containing the MD simulation parameters: (initial T, final T, time step in fs)
traj = Trajectory("lammps",(lammps_dump_path, MDtup))


# calculate structural dynamics, here are the defaults
traj.dynamics(   # general parameters
                 read_mode, # key parameter, 1: static mode, 2: transient mode, no default
                 uniname = "test", # A unique user-defined name for this trajectory, will be used in printing and figure saving
                 allow_equil = 0.5, # take the first x fraction of the trajectory as equilibration, this part will not be computed
                 read_every = 0, # read only every n steps, default is 0 which the code will decide an appropriate value according to the system size
                 coords_time_average = 0, # time-averaging of coordinates, input t>0 as the average time window with the unit of picosecond. Use with caution. 
                 saveFigures = False, # whether to save produced figures
                 lib_saver = False,  # whether to save computed material properties in lib file
                 lib_overwrite = False, # whether to overwrite existing lib entry, or just change upon them
	
	         # manually define system info that is saved in the class template
                 system_overwrite = None, # dict contains X-site and B-site info, and the default bond lengths, see README
                 
                 # function toggles
                 preset = 0, # 0: no preset, uses the toggles, 1: lat & tilt_distort, 2: lat & tilt_distort & tavg & MO, 3: all
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
                 
                 # time averaged structure
                 start_ratio = None, # time-averaging structure ratio, e.g. 0.9 means only averaging the last 10% of trajectory, default = allow_equil
                 Asite_reconstruct = False, # setting a different time-averaging algo for organic A-sites
                 
                 # octahedral tilting and distortion
                 structure_type = 1, # 1: 3C polytype, 2: other non-perovskite with orthogonal reference enabled, 3: other non-perovskite with initial config as reference. Please note that mode 2 and 3 are relatively less tested.   
                 multi_thread = 1, # if >1, enable multi-threading in this calculation, since not vectorized
                 rotation_from_orthogonal = None, # None: code will detect if the BX6 frame is not orthogonal to the principle directions, only manually input this [x,y,z] rotation angles in degrees if told by the code. 
                 tilt_corr_NN1 = True, # enable first NN correlation of tilting, reflecting the Glazer notation
                 full_NN1_corr = False, # include off-diagonal correlation terms 
                 tilt_corr_spatial = False, # enable spatial correlation beyond NN1
                 tiltautoCorr = False, # compute Tilting decorrelation time constant
                 enable_refit = False, # refit the octahedral network in case of change of geometry
                 symm_n_fold = 0, # tilting range, 0: auto, 2: [-90,90], 4: [-45,45], 8: [0,45]
                 tilt_recenter = False, # whether to eliminate the shift in tilting values according to the mean value of population
                 tilt_domain = False, # compute the time constant of tilt correlation domain formation
                 
                 # molecular orientation (MO)
                 MOautoCorr = False, # compute MO reorientation time constant
                 MO_corr_spatial = False, # enable spatial correlation function of MO
                 )
