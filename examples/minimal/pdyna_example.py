from pdyna.core import Trajectory

traj = Trajectory("vasp",("POSCAR_mapi_example", "XDATCAR_mapi_example", "INCAR_mapi_example"))

traj.dynamics(uniname="any_name_you_like", # A unique user-defined name for this trajectory, will be used in printing and figure saving
             read_mode= 1, # key parameter, 1: equilibration mode, 2: quench/anneal mode
             preset = 2, # lat & tilt_distort & tavg & MO
             lat_method = 2, # enable pseudo-cubic lattice parameter 
             )
