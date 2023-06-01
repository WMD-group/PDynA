# choose an initialization with respect to your trajectory type
# VASP XDATCAR
traj = Trajectory("vasp",(poscar_path, xdatcar_path, incar_path))

# or LAMMPS dump file
MDtup = (100,100,1) # tuple containing the MD simulation parameters: (initial T, final T, time step in fs)
traj = Trajectory("lammps",(lammps_dump_path, MDtup))


# calculate structural dynamics, here are the defaults
traj.dynamics(read_mode = no_default,      # key parameter, 1: equilibration mode (time-independent property computation), 2: transient mode (time/temperature-dependent properties)
                  uniname= 'test',         # A unique user-defined name for this trajectory, will be used in printing and figure/structure saving
                  allow_equil = 0.5,       # take the first x fraction of the trajectory as equilibration, this part will not be computed (only used in read_mode 1)
                  read_every = 0,          # read only every n steps, default is 0 which the code will decide an appropriate value according to the system size
                  saveFigures = False,     # whether to save produced figures
                  lib_saver = False,       # whether to save computed material properties in lib file, a file named 'perovskite_gaussian_data' will be generated with Pickle
                  lib_overwrite = False,   # whether to overwrite existing lib entry (True), or just change those that are computed in this run (False)

                  # function toggles
                  preset = 2,                  # 0: no preset, uses the toggles, 1: lat & tilt_distort, 2: lat & tilt_distort & tavg & MO, 3: all
                  toggle_lat = False,          # switch of lattice parameter calculation
                  toggle_tavg = False,         # switch of time averaged structure
                  toggle_tilt_distort = False, # switch of octahedral tilting and distortion calculation
                  toggle_MO = False,           # switch of molecular orientation (MO) calculation (for organic A-site)
                  toggle_RDF = False,          # switch of radial distribution function calculation
                  toggle_A_disp = False,       # switch of A-site cation displacement calculation 
		  
                  smoother = 0,  # whether to use S-G smoothing on outputs (used in read_mode 2), 0: disabled, >0: average window in ps

                  # Lattice parameter calculation
                  lat_method = 1,         # lattice parameter analysis methods, 1: direct lattice cell dimension, 2: pseudo-cubic lattice parameter
                  zdir = 2,               # specified z-direction in case of lat_method 2
                  lattice_rot = [0,0,0],  # the rotation of system in prior to lattice calculation in case of lat_method 2

                  # time averaged structure
                  start_ratio = 0.5,          # time-averaging structure ratio, e.g. 0.9 means only averaging the last 10% of trajectory
                  tavg_save_dir = '.\\',      # directory for saving the time-averaging structures 
                  Asite_reconstruct = False,  # setting a different time-averaging algo for organic A-sites

                  # octahedral tilting and distortion
                  structure_type = 1,         # 1: 3C polytype, 2: other non-perovskite with orthogonal reference enabled, 3: other non-perovskite with initial config as reference             
                  multi_thread = 1,           # if >1, enable multi-threading in this calculation with n threads
                  tilt_corr_NN1 = True,       # enable first NN correlation of tilting, reflecting the Glazer notation
                  full_NN1_corr = False,      # include off-diagonal correlation terms 
                  tilt_corr_spatial = False,  # enable spatial correlation beyond NN1
                  tiltautoCorr = False,       # compute Tilting decorrelation time constant (experimental)
                  octa_locality = False,      # compute differentiated properties within mixed-halide sample
                  enable_refit = False,       # refit the octahedral network in case of change of geometry (multi_thread will be disabled)
                  symm_n_fold = 0,            # tilting range, 0: auto, 2: [-90,90], 4: [-45,45], 8: [0,45]
                  tilt_domain = False,        # compute the time constant of tilt correlation domain formation
                  vis3D_domain = 0,           # 3D visualization of tilt domain in time. 0: off, 1: tilt angle, 2: tilting correlation polarity
                  
                  
                  # molecular orientation (MO)
                  MOautoCorr = False,       # compute MO reorientation time constant
                  MO_corr_spatial = False,  # enable first and second NN correlation function of MO
                  draw_MO_anime = False,    # plot the MO in 3D animation, will take a few minutes
                  )