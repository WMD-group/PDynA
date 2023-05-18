from pdyna.core import Trajectory

MDtup = (100,100,1) # tuple containing the MD simulation parameters: (initial T, final T, time step in fs)
traj = Trajectory("lammps",('lammps_example_mapbbr3.out', MDtup))

traj.dynamics(read_mode= 1, # key parameter, 1: equilibration mode, 2: quench/anneal mode
              uniname='test', # A unique user-defined name for this trajectory, will be used in printing and figure saving
              allow_equil = 0.5, # take the first x fraction of the trajectory as equilibration, this part will not be computed
              read_every = 0, # read only every n steps, default is 0 which the code will decide an appropriate value according to the system size
              saveFigures = True, # whether to save produced figures

              # function toggles
              preset = 2, # 0: no preset, uses the toggles, 1: lat & tilt_distort, 2: lat & tilt_distort & tavg & MO, 3: all
              
              # Lattice parameter calculation
              lat_method = 2, # lattice parameter analysis methods, 1: direct lattice cell dimension, 2: pseudo-cubic lattice parameter
              zdir = 1, # specified z-direction in case of lat_method 2

              # time averaged structure
              start_ratio = 0.8, # time-averaging structure ratio, e.g. 0.9 means only averaging the last 10% of trajectory
              tavg_save_dir = '.\\', # directory for saving the time-averaging structures 

              # octahedral tilting and distortion
              structure_type = 1, # 1: 3C polytype, 2: other non-perovskite with orthogonal reference enabled, 3: other non-perovskite with initial config as reference             
              multi_thread = 8, # enable multi-threading in this calculation, since not vectorized
              )