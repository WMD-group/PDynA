"""
pdyna.structural: The collection of structural processing functions.

"""
import pdyna
import contextlib
import numpy as np
import os, json, sys
import matplotlib.pyplot as plt
from pymatgen.core.operations import SymmOp
from pymatgen.core.structure import Molecule
from pymatgen.core.periodic_table import Element
from pymatgen.analysis.molecule_matcher import HungarianOrderMatcher


# loading the octahedral basis files
filename_basis = os.path.join(os.path.abspath(pdyna.__file__).rstrip('__init__.py'),'basis/octahedron_basis_full.json')
try:
    with open(filename_basis, 'r') as f:
        dict_basis = json.load(f)
        dist_dim = len(dict_basis)-3
except IOError:
    sys.stderr.write('IOError: failed reading from {}.'.format(filename_basis))
    sys.exit(1)
    
@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """
    Supportive function for adding progress bar within joblib parallelized computation. 
    """
    import joblib
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


def traj_time_average(Allpos,latmat,MDTimestep,twindow):
    """
    Compute the time-averaged structure from coordinates.
    
    Args:   
        Allpos (numpy.ndarray): Coordinate array of all atoms.
        latmat (numpy.ndarray): Lattice matrix of all frames, with a shape of (N, 3, 3).
        MDTimestep (float): Time step of the MD simulation.
        twindow (float): Time window for averaging.
    
    Returns: 
        numpy.ndarray: Time-averaged coordinates.
        numpy.ndarray: Time-averaged lattice matrix.
        int: Number of atoms in the time-averaged structure.
    """
    avgfr = round(twindow/MDTimestep)
    fracs = get_frac_from_cart(Allpos,latmat)
    ftavg = np.empty((fracs.shape[0]-avgfr+1,fracs.shape[1],3))
    for i in range(fracs.shape[1]):
        for j in range(3):
            ftavg[:,i,j] = np.convolve(fracs[:,i,j],np.ones(avgfr)/avgfr,'valid')
    lmat = latmat.copy()
    lavg = np.empty((fracs.shape[0]-avgfr+1,3,3))
    for i in range(3):
        for j in range(3): 
            lavg[:,i,j] = np.convolve(lmat[:,i,j],np.ones(avgfr)/avgfr,'valid')
    
    # store into object attr
    ctavg = get_cart_from_frac(ftavg,lavg)
    timeori = np.linspace(0,fracs.shape[0]-1,fracs.shape[0])*MDTimestep
    timeavg = np.convolve(timeori,np.ones(avgfr)/avgfr,'valid')
    
    fig,ax = plt.subplots()
    plt.plot(timeori,Allpos[:,0,0],alpha=0.8,label='raw')
    plt.plot(timeavg,ctavg[:,0,0],alpha=0.8,label='t-averaged')
    plt.title("Time averaging: x-coords of atom 0")
    plt.xlabel("time (ps)")
    plt.ylabel("X coordinate (Angstrom)")
    plt.legend()
    plt.show()
    
    return ctavg, lavg, ctavg.shape[0]


def resolve_octahedra(Bpos,Xpos,readfr,at0,enable_refit,multi_thread,latmat,fpg_val_BX,neigh_list,orthogonal_frame,structure_type,complex_pbc,ref_initial=None,rtr=None):
    """
    Compute octahedral tilting and distortion from the trajectory coordinates and octahedral connectivity.

    Args:
        Bpos (numpy.ndarray): Coordinate array of B-sites.
        Xpos (numpy.ndarray): Coordinate array of X-sites.
        readfr (list): List of frame numbers to be computed.
        at0 (ASE Atoms): Atoms object of the initial frame.
        enable_refit (bool): Enabling of structure refitting during computation if a large distortion is found.
        multi_thread (bool): Enable multi-threading for computation.
        latmat (numpy.ndarray): Lattice matrix of all frames, with a shape of (N, 3, 3).
        fpg_val_BX (list): Defined B-X bond information at the beginning of the class.
        neigh_list (numpy.ndarray): The B-X connectivity matrix of octahedra with shape (N, 6).
        orthogonal_frame (bool): If the structure is 3D perovskite and aligned in the orthogonal directions.
        structure_type (int): Structure type indicating orientation and connectivity of octahedra.
        complex_pbc (bool): If the cell has strong tilts.
        ref_initial (numpy.ndarray, optional): For non-orthogonal structure, the individual reference for each octahedron.
        rtr (numpy.ndarray, optional): A (3, 3) rotation matrix from orthogonal directions if specified.

    Returns:
        numpy.ndarray: Computed array of octahedral distortion with a shape of (N, dist_dim).
        numpy.ndarray: Computed array of octahedral tilting with a shape of (N, 3).
        numpy.ndarray: Record of structural refitting if parameter enable_refit is True, shape is (N, 2), giving a frame number and a bool indicating if refit is performed.
    """
    from tqdm import tqdm
    from scipy.spatial.transform import Rotation as sstr
    
    def frame_wise(fr):
        """
        Compute octahedral tilting and distortion for a single frame, can be parallelized with multi-threading.
        """
        mymat=latmat[fr,:]
        
        disto = np.empty((0,dist_dim))
        Rmat = np.zeros((Bcount,3,3))
        Rmsd = np.zeros((Bcount,1))
        for B_site in range(Bcount): # for each B-site atom
            if np.isnan(neigh_list[B_site,:]).any(): 
                a = np.empty((1,dist_dim))
                a[:] = np.nan
                Rmat[B_site,:] = np.nan
                Rmsd[B_site] = np.nan
                disto = np.concatenate((disto,a),axis = 0)
            else:
                raw = Xpos[fr,neigh_list[B_site,:].astype(int),:] - Bpos[fr,B_site,:]
                bx = octahedra_coords_into_bond_vectors(raw,mymat)
                if not orthogonal_frame:
                    bx = np.matmul(bx,ref_initial[B_site,:])
                if not rtr is None:
                    bx = np.matmul(bx,rtr)
          
                dist_val,rotmat,rmsd = calc_distortions_from_bond_vectors_full(bx) # _full
                    
                Rmat[B_site,:] = rotmat
                Rmsd[B_site] = rmsd
                disto = np.concatenate((disto,dist_val.reshape(1,dist_dim)),axis = 0)
        
        tilts = np.zeros((Bcount,3))
        for i in range(Rmat.shape[0]):
            if np.isnan(neigh_list[i,:]).any(): 
                tilts[i,:] = np.nan
            else:
                tilts[i,:] = sstr.from_matrix(Rmat[i,:]).as_euler('xyz', degrees=True)

        #tilts = periodicity_fold(tilts) 
        return (disto.reshape(1,Bcount,dist_dim),tilts.reshape(1,Bcount,3),Rmsd)
    
    if (not ref_initial is None) and (not rtr is None):
        raise TypeError("individual reference of octahedra and orthogonal lattice alignment are simulaneously enabled. Check structure_type and rotation_from_orthogonal parameters. ")

    Bcount = Bpos.shape[1]
    refits = np.empty((0,2))
    
    #if not rotation_from_orthogonal is None:
    #    rtr = np.linalg.inv(rotation_from_orthogonal)
    
    if multi_thread == 1: # multi-threading disabled and can do in-situ refit
        
        Di = np.empty((len(readfr),Bcount,dist_dim))
        T = np.empty((len(readfr),Bcount,3))
        
        for subfr in tqdm(range(len(readfr))):
            fr = readfr[subfr]
            mymat=latmat[fr,:]
            
            disto = np.empty((0,dist_dim))
            Rmat = np.zeros((Bcount,3,3))
            Rmsd = np.zeros((Bcount,1))
            for B_site in range(Bcount): # for each B-site atom
                if np.isnan(neigh_list[B_site,:]).any(): 
                    a = np.empty((1,dist_dim))
                    a[:] = np.nan
                    Rmat[B_site,:] = np.nan
                    Rmsd[B_site] = np.nan
                    disto = np.concatenate((disto,a),axis = 0)
                else:
                    raw = Xpos[fr,neigh_list[B_site,:].astype(int),:] - Bpos[fr,B_site,:]
                    bx = octahedra_coords_into_bond_vectors(raw,mymat)
                    if not orthogonal_frame:
                        bx = np.matmul(bx,ref_initial[B_site,:])
                    if not rtr is None:
                        bx = np.matmul(bx,rtr)
         
                    dist_val,rotmat,rmsd = calc_distortions_from_bond_vectors_full(bx)

                    Rmat[B_site,:] = rotmat
                    Rmsd[B_site] = rmsd
                    disto = np.concatenate((disto,dist_val.reshape(1,dist_dim)),axis = 0)
            
            if enable_refit and fr > 0 and max(np.amax(disto)) > 1: # re-fit neigh_list if large distortion value is found
                disto_prev = np.nanmean(disto,axis=0)
                neigh_list_prev = neigh_list
                
                Bpos_frame = Bpos[fr,:]     
                Xpos_frame = Xpos[fr,:]
                rt = distance_matrix_handler(Bpos_frame,Xpos_frame,latmat[fr,:],at0.cell,at0.pbc,complex_pbc)
                if orthogonal_frame:
                    if rtr is None:
                        neigh_list = fit_octahedral_network_defect_tol(Bpos_frame,Xpos_frame,rt,mymat,fpg_val_BX,structure_type)
                    else: # non-orthogonal
                        neigh_list = fit_octahedral_network_defect_tol_non_orthogonal(Bpos_frame,Xpos_frame,rt,mymat,fpg_val_BX,structure_type,np.linalg.inv(rtr))
                else:
                    neigh_list, ref_initial = fit_octahedral_network_defect_tol(Bpos_frame,Xpos_frame,rt,mymat,fpg_val_BX,structure_type)

                mymat=latmat[fr,:]
                
                # re-calculate distortion values
                disto = np.empty((0,dist_dim))
                Rmat = np.zeros((Bcount,3,3))
                Rmsd = np.zeros((Bcount,1))
                for B_site in range(Bcount): # for each B-site atom
                        
                    raw = Xpos[fr,neigh_list[B_site,:].astype(int),:] - Bpos[fr,B_site,:]
                    bx = octahedra_coords_into_bond_vectors(raw,mymat)
                    if not orthogonal_frame:
                        bx = np.matmul(bx,ref_initial[B_site,:])
                    if not rtr is None:
                        bx = np.matmul(bx,rtr)
                    
                    dist_val,rotmat,rmsd = calc_distortions_from_bond_vectors_full(bx)
                        
                    Rmat[B_site,:] = rotmat
                    Rmsd[B_site] = rmsd
                    disto = np.concatenate((disto,dist_val.reshape(1,dist_dim)),axis = 0)
                if np.array_equal(np.nanmean(disto,axis=0),disto_prev) and np.array_equal(neigh_list,neigh_list_prev):
                    refits = np.concatenate((refits,np.array([[fr,0]])),axis=0)
                else:
                    refits = np.concatenate((refits,np.array([[fr,1]])),axis=0)
            
            Di[subfr,:,:] = disto.reshape(1,Bcount,dist_dim)
            
            tilts = np.zeros((Bcount,3))
            for i in range(Rmat.shape[0]):
                tilts[i,:] = sstr.from_matrix(Rmat[i,:]).as_euler('xyz', degrees=True) 
            #tilts = periodicity_fold(tilts)
            T[subfr,:,:] = tilts.reshape(1,Bcount,3)
            
            
    elif multi_thread > 1: # multi-threading enabled and cannot refit
        from joblib import Parallel, delayed
        Di = np.empty((len(readfr),Bcount,dist_dim))
        T = np.empty((len(readfr),Bcount,3))
        
        with tqdm_joblib(tqdm(desc="Progress", total=len(readfr))) as progress_bar:
            results = Parallel(n_jobs=multi_thread)(delayed(frame_wise)(fr) for fr in readfr)
            
        # unpack the result from multi-threading calculation
        assert len(results) == len(readfr)
        for fr, each in enumerate(results):
            Di[fr,:,:] = each[0]
            T[fr,:,:] = each[1]
    
    else:
        raise ValueError("The input multi-threading count must be a positive integer.")
        
    
    return Di, T, refits


def fit_octahedral_network_frame(Bpos_frame,Xpos_frame,r,mymat,fpg_val_BX,rotated,rotmat):
    """ 
    Resolve the octahedral connectivity. 

    Args:
        Bpos_frame (numpy.ndarray): Coordinate array of B-sites.
        Xpos_frame (numpy.ndarray): Coordinate array of X-sites.
        r (numpy.ndarray): Distance matrix of B-X sites.
        mymat (numpy.ndarray): Lattice matrix of the frame.
        fpg_val_BX (list): Defined B-X bond information at the beginning of the class.
        rotated (bool): If the structure is rotated.
        rotmat (numpy.ndarray): Rotation matrix of the structure to align with orthogonal directions.

    Returns:
        numpy.ndarray: The B-X connectivity matrix of octahedra with shape (N, 6). 
    """

    max_BX_distance = find_population_gap(r, fpg_val_BX[0], fpg_val_BX[1])
        
    neigh_list = np.zeros((Bpos_frame.shape[0],6))
    for B_site, X_list in enumerate(r): # for each B-site atom
        X_idx = [i for i in range(len(X_list)) if X_list[i] < max_BX_distance]
        if len(X_idx) != 6:
            raise ValueError(f"The number of X site atoms connected to B site atom no.{B_site} is not 6 but {len(X_idx)}. \n")
        
        bx_raw = Xpos_frame[X_idx,:] - Bpos_frame[B_site,:]
        bx = octahedra_coords_into_bond_vectors(bx_raw,mymat)
        
        if rotated:
            order1 = match_bx_orthogonal_rotated(bx,rotmat) 
        else:
            order1 = match_bx_orthogonal(bx) 
        neigh_list[B_site,:] = np.array(X_idx)[order1]

    
    neigh_list = neigh_list.astype(int)

    return neigh_list



def fit_octahedral_network_defect_tol(Bpos_frame,Xpos_frame,r,mymat,fpg_val_BX,structure_type):
    """ 
    Resolve the octahedral connectivity with a defect tolerance. 

    Args:
        Bpos_frame (numpy.ndarray): Coordinate array of B-sites.
        Xpos_frame (numpy.ndarray): Coordinate array of X-sites.
        r (numpy.ndarray): Distance matrix of B-X sites.
        mymat (numpy.ndarray): Lattice matrix of the frame.
        fpg_val_BX (list): Defined B-X bond information at the beginning of the class.
        structure_type (int): Structure type indicating orientation and connectivity of octahedra.

    Returns:    
        numpy.ndarray: The B-X connectivity matrix of octahedra with shape (N, 6).
    """
    
    try:
        max_BX_distance = find_population_gap(r, fpg_val_BX[0], fpg_val_BX[1], tol=5) # tol is set for extrame cases with defects
        
        bxs = []
        bxc = []
        for B_site, X_list in enumerate(r): # for each B-site atom
            X_idx = [i for i in range(len(X_list)) if X_list[i] < max_BX_distance]
            bxc.append(len(X_idx))
            bxs.append(X_idx)
        bxc = np.array(bxc)
        
        ndef = None
        if np.amax(bxc) != 6 or np.amin(bxc) != 6:
            ndef = np.sum(bxc!=6) 
            if np.amax(bxc) < 8 and np.amin(bxc) > 4: # benign defects
                print(f"!Structure Resolving: the initial structure contains {ndef} out of {len(bxc)} octahedra with defects. The algorithm has screened them out of the population. ")
            else: # bad defects
                raise TypeError("!Structure Resolving: the initial structure contains octahedra with complex defect configuration, leading to unresolvable connectivity. ")
    
    except (ValueError,TypeError):
        closebx = np.argsort(r, axis=0)[:2, :]
        binx, bcount = np.unique(closebx, return_counts=True)
        if (not np.array_equal(binx,np.arange(0,r.shape[0]))) or (not np.array_equal(bcount,np.ones_like(bcount)*6)):
            raise ValueError("!Structure resolving: Can't separate the different neighbours, please check the fpg_val values are correct according to the guidance at the beginning of the Trajectory class, or check if initial structure is defected or too distorted. ")
        #print("!Sructure Resolving: Detected relative high deviation in the B-X connectivity, used a more aggresive fit, please monitor the outputs carefully. ")
        bxs = convert_xb_to_bx(closebx)
           
    neigh_list = np.zeros((Bpos_frame.shape[0],6))
    ref_initial = np.zeros((Bpos_frame.shape[0],3,3))
    #bxfit_rmsd = []
    for B_site, bxcom in enumerate(bxs): # for each B-site atom
        if len(bxcom) == 6:
            #rmsd = None
            try: # remove this try for strict octahedral match
                bx_raw = Xpos_frame[bxcom,:] - Bpos_frame[B_site,:]
                bx = octahedra_coords_into_bond_vectors(bx_raw,mymat)
                
                if structure_type == 1:
                    order1 = match_bx_orthogonal(bx) 
                    neigh_list[B_site,:] = np.array(bxcom)[order1].astype(int)
                elif structure_type == 2:
                    order1, _, rmsd = match_bx_arbitrary(bx) 
                    neigh_list[B_site,:] = np.array(bxcom)[order1].astype(int)
                elif structure_type == 3:   
                    order1, ref1, rmsd = match_bx_arbitrary(bx) 
                    neigh_list[B_site,:] = np.array(bxcom)[order1].astype(int)
                    ref_initial[B_site,:] = ref1
                #bxfit_rmsd.append(rmsd)
            
            except ValueError: 
                if structure_type in (1,2):
                    neigh_list[B_site,:] = np.nan
                else:   
                    neigh_list[B_site,:] = np.nan
                    ref_initial[B_site,:] = np.nan # til here

        else:
            if structure_type in (1,2):
                neigh_list[B_site,:] = np.nan
            else:   
                neigh_list[B_site,:] = np.nan
                ref_initial[B_site,:] = np.nan
    
    if np.sum(np.isnan(neigh_list[:,0])) > neigh_list.shape[0]*0.1 and np.sum(np.isnan(neigh_list[:,0]))>1:
        raise ValueError(f"There are {np.sum(np.isnan(neigh_list[:,0]))} out of {neigh_list.shape[0]} octahedra contains unsolvable B-X connections.")
    #elif np.sum(np.isnan(neigh_list[:,0])) != 0:
    #    print(f"!Structural Resolving: There are {np.sum(np.isnan(neigh_list[:,0]))} out of {neigh_list.shape[0]} octahedra contains unsolvable B-X connections.")
    
    if structure_type in (1,2):
        return neigh_list
    else:
        return neigh_list, ref_initial
    

def fit_octahedral_network_defect_tol_non_orthogonal(Bpos_frame,Xpos_frame,r,mymat,fpg_val_BX,structure_type,rotmat):
    """ 
    Resolve the octahedral connectivity with a defect tolerance. 
    
    Args:
        Bpos_frame (numpy.ndarray): Coordinate array of B-sites.
        Xpos_frame (numpy.ndarray): Coordinate array of X-sites.
        r (numpy.ndarray): Distance matrix of B-X sites.
        mymat (numpy.ndarray): Lattice matrix of the frame.
        fpg_val_BX (list): Defined B-X bond information at the beginning of the class.
        structure_type (int): Structure type indicating orientation and connectivity of octahedra.
        rotmat (numpy.ndarray): Rotation matrix of the structure to align with orthogonal directions.

    Returns:
        numpy.ndarray: The B-X connectivity matrix of octahedra with shape (N, 6).
    """
    
    if structure_type != 1:
        raise TypeError("The non-orthogonal initial structure can only be analysed under the 3C (corner-sharing) connectivity. ")
    #rotmat = np.linalg.inv(rotmat)
        
    try:
        max_BX_distance = find_population_gap(r, fpg_val_BX[0], fpg_val_BX[1], tol=5)
        
        bxs = []
        bxc = []
        for B_site, X_list in enumerate(r): # for each B-site atom
            X_idx = [i for i in range(len(X_list)) if X_list[i] < max_BX_distance]
            bxc.append(len(X_idx))
            bxs.append(X_idx)
        bxc = np.array(bxc)
        
        ndef = None
        if np.amax(bxc) != 6 or np.amin(bxc) != 6:
            ndef = np.sum(bxc!=6) 
            if np.amax(bxc) == 7 and np.amin(bxc) == 5: # benign defects
                print(f"!Structure Resolving: the initial structure contains {ndef} out of {len(bxc)} octahedra with defects. The algorithm has screened them out of the population. ")
            else: # bad defects
                raise TypeError("!Structure Resolving: the initial structure contains octahedra with complex defect configuration, leading to unresolvable connectivity. ")
    
    except (ValueError,TypeError):
        closebx = np.argsort(r, axis=0)[:2, :]
        binx, bcount = np.unique(closebx, return_counts=True)
        if (not np.array_equal(binx,np.arange(0,r.shape[0]))) or (not np.array_equal(bcount,np.ones_like(bcount)*6)):
            raise ValueError("!Structure resolving: Can't separate the different neighbours, please check the fpg_val values are correct according to the guidance at the beginning of the Trajectory class, or check if initial structure is defected or too distorted. ")
        print("!Sructure Resolving: Detected relatively high deviation in the B-X connectivity, used a more aggresive fit, please monitor the outputs carefully. ")
        bxs = convert_xb_to_bx(closebx)
        
    neigh_list = np.zeros((Bpos_frame.shape[0],6))
    ref_initial = np.zeros((Bpos_frame.shape[0],3,3))
    for B_site, bxcom in enumerate(bxs): # for each B-site atom
        if len(bxcom) == 6:
            bx_raw = Xpos_frame[bxcom,:] - Bpos_frame[B_site,:]
            bx = octahedra_coords_into_bond_vectors(bx_raw,mymat)

            order1 = match_bx_orthogonal_rotated(bx,rotmat) 
            neigh_list[B_site,:] = np.array(bxcom)[order1].astype(int)

        else:
            neigh_list[B_site,:] = np.nan

    return neigh_list


def convert_xb_to_bx(xb):
    """
    Convert the X-to-B connectivity matrix to the ordinary B-to-X connectivity matrix.

    Args:
        xb (numpy.ndarray): The X-to-B connectivity matrix of octahedra.
        
    Returns:
        numpy.ndarray: The B-X connectivity matrix with shape (N, 6).
    """
    
    bx = [[] for i in range(int(xb.shape[1]/3))]
    for i in range(xb.shape[1]):
        bx[xb[0,i]].append(i)
        bx[xb[1,i]].append(i)
    bx = np.array(bx)
    assert bx.shape==(int(xb.shape[1]/3),6)
    
    return bx


def find_polytype_network(Bpos_frame,Xpos_frame,r,mymat,neigh_list):
    """ 
    Resolve the octahedral connectivity and output the polytype information. 

    Args:   
        Bpos_frame (numpy.ndarray): Coordinate array of B-sites.
        Xpos_frame (numpy.ndarray): Coordinate array of X-sites.
        r (numpy.ndarray): Distance matrix of B-X sites.
        mymat (numpy.ndarray): Lattice matrix of the frame.
        neigh_list (numpy.ndarray): The B-X connectivity matrix of octahedra with shape (N, 6).

    Returns:
        list: List of strings indicating the local polytype connecivity of each octahedron.
        list: List of arrays indicating the shared X-sites with neighbouring B-sites.
        dict: Dictionary of octahedra categories.
    """
    
    if np.isnan(neigh_list).any():
        raise TypeError("Polytype recognition is not compatible with initial structure with defects at the moment.")
    
    def category_register(cat,typestr,b):
        """
        Register the octahedra into categories dict.
        """
        if not typestr in cat:
            cat[typestr] = []
        cat[typestr].append(b)
        return cat

    BBsearch = 10.0  # 8.0

    connectivity = []
    conntypeStr = []
    conntypeStrAll = []
    conn_category = {}
    #plt.hist(dm.reshape(-1,),bins=100,range=[0,15])
    for B0, B1_list in enumerate(r): # for each B-site atom
        B1 = [i for i in range(len(B1_list)) if (B1_list[i] < BBsearch and i != B0)]
        if len(B1) == 0:
            raise ValueError(f"Can't find any other B-site around B-site number {B0} within a search radius of {BBsearch} angstrom. ")
            
        conn = np.empty((0,2))
        for b1 in B1:
            intersect = list(set(neigh_list[B0,:]).intersection(neigh_list[b1,:]))
            if len(intersect) > 0:
                if len(intersect) == 2: #consider special case PBC effect for case 'intersect 2'
                    bond_raw = Xpos_frame[intersect,:] - Bpos_frame[B0,:]
                    bond0 = apply_pbc_cart_vecs(bond_raw, mymat)
                    bond_raw = Xpos_frame[intersect,:] - Bpos_frame[b1,:]
                    bond1 = apply_pbc_cart_vecs(bond_raw, mymat)
                    dots = np.sum((bond0/np.linalg.norm(bond0,axis=1).reshape(2,1))*(bond1/np.linalg.norm(bond1,axis=1).reshape(2,1)), axis=1)
                    if np.amax(dots) < -0.8: # effectively cornor-sharing
                        intersect = [intersect[0]]
                conn = np.concatenate((conn,np.array([[b1,len(intersect)]]))).astype(int)
                
        if conn.shape[0] == 0:
            conntypeStr.append("isolated")
            conntypeStrAll.append("isolated")
            connectivity.append(conn)
            conn_category = category_register(conn_category, "isolated", B0)
            
        else:
            conntype = set(list(conn[:,1]))
            if len(conntype-{1,2,3}) != 0:
                raise TypeError(f"Found an unexpected connectivity type {conntype}. ")
            
            if len(conntype) == 1: # having only one connectivity type
                if conntype == {1}:
                    conntypeStr.append("corner")
                    conntypeStrAll.append("corner")
                    conn_category = category_register(conn_category, "corner", B0)
                elif conntype == {2}:
                    conntypeStr.append("edge")
                    conntypeStrAll.append("edge")
                    conn_category = category_register(conn_category, "edge", B0)
                elif conntype == {3}:
                    conntypeStr.append("face")
                    conntypeStrAll.append("face")
                    conn_category = category_register(conn_category, "face", B0)
            else:
                strt = []
                for ct in conntype:
                    if ct == 1:
                        strt.append("corner")
                        conntypeStrAll.append("corner")
                    elif ct == 2:
                        strt.append("edge")
                        conntypeStrAll.append("edge")
                    elif ct == 3:
                        strt.append("face")
                        conntypeStrAll.append("face")
                strt = "+".join(strt)
                conntypeStr.append(strt)
                conn_category = category_register(conn_category, strt, B0)
                
            connectivity.append(conn)
        
    if len(set(conntypeStr)) == 1:
        print(f"Octahedral connectivity: {list(set(conntypeStr))[0]}-sharing")
    else:
        conntypestr = "+".join(list(set(conntypeStrAll)))
        print(f"Octahedral connectivity: mixed - {conntypestr}")    
        
    return conntypeStr, connectivity, conn_category
        

def pseudocubic_lat(traj,  # the main class instance
                    allow_equil = 0, # take the first x fraction of the trajectory as equilibration
                    zdrc = 2,  # zdrc: the z direction for pseudo-cubic lattice paramter reading. a,b,c = 0,1,2
                    lattice_tilt = None, # if primary B-B bond is not along orthogonal directions, in degrees
                    ): 
    
    """ 
    Compute the pseudo-cubic lattice parameters. 

    Args:
        traj (Trajectory): The main class instance.
        allow_equil (float): The fraction of the trajectory to be considered as equilibration.
        zdrc (int): The z direction for pseudo-cubic lattice paramter reading. a,b,c = 0,1,2.
        lattice_tilt (numpy.ndarray): If the primary B-B bond is not along orthogonal directions, in degrees.

    Returns:
        numpy.ndarray: The pseudo-cubic lattice parameters.
    """
               
    from scipy.stats import norm
    from scipy.stats import binned_statistic_dd as binstat
    
    def run_pcl(zi,):
        
        dims = [0,1,2]
        dims.remove(zi)
        grided = bin_indices[dims,:].T
        
        ma = np.array([[[0,1,1],[0,1,-1],[2,0,0]],
                       [[-1,0,1],[1,0,1],[0,2,0]],
                       [[-1,1,0],[1,1,0],[0,0,2]]]) # setting of pseuso-cubic lattice parameter vectors
        
        mappings = BBdist*ma
        maps = mappings[zi,:,:] # select out of x,y,z
        maps_frac = ma[zi,:,:]
        
        if not lattice_tilt is None:
            maps = np.dot(maps,lattice_tilt)
        
        match_tol = 1.5 # tolerance of the coordinate difference is set to 1.5 angstroms    
        
        if len(blist) <= 8:
            Im = np.empty((0,3))
            for i in range(-1,2): # create images of all surrounding cells
                for j in range(-1,2):
                    for k in range(-1,2):
                        Im = np.concatenate((Im,np.array([[i,j,k]])),axis=0) 
            
            Bfc = st0.frac_coords[blist,:] # frac_coords of B site atoms

            Bnet = np.zeros((Bfc.shape[0],3,2)) # find the corresponding atoms (with image) for lattice parameter calc.
            # dim0: for each B atom, dim1: a,b,c vector of that atom, dim2: [B atom number, image]
            Bnet[:] = np.nan
            for B1 in range(Bfc.shape[0]):
                for im in range(Im.shape[0]):
                    for B2 in range(Bfc.shape[0]):
                        vec = st0.lattice.get_cartesian_coords(Im[im,:] + Bfc[B1,:] - Bfc[B2,:])
                        for i in range(3):
                            if np.linalg.norm(vec-maps[i,:]) < match_tol: 

                                if np.isnan(Bnet[B1,i,0]) and np.isnan(Bnet[B1,i,1]):
                                    Bnet[B1,i,0] = B2
                                    Bnet[B1,i,1] = im
                                else:
                                    raise ValueError("A second fitted neighbour is written to the site. Decrease match_tol")
                        
            if np.isnan(Bnet).any():
                raise TypeError("Some of the neightbours are not found (still nan). Increase match_tol")
            Bnet = Bnet.astype(int)
            
            # find the correct a and b vectors
            #code = np.remainder(np.sum(grided,axis=1),2)
            code = np.remainder(grided[:,1],2)
            
            Bfrac = get_frac_from_cart(traj.Allpos[:,traj.Bindex,:],traj.latmat)
            lats = traj.latmat
            
            Lati_on = np.zeros((len(range(round(traj.nframe*allow_equil),traj.nframe)),Bfc.shape[0],3))
            
            for ite, fr in enumerate(range(round(traj.nframe*allow_equil),traj.nframe)):
                Bfc = Bfrac[fr,:,:]
                templat = np.empty((Bfc.shape[0],3))
                for B1 in range(Bfc.shape[0]):
                    for i in range(3):
                        if i == 2: # identified z-axis
                            templat[B1,i] = np.linalg.norm(np.dot((Im[Bnet[B1,i,1],:] + Bfc[B1,:] - Bfc[Bnet[B1,i,0]]),lats[fr,:,:]))/2
                        else: # x- and y-axis
                            temp = np.linalg.norm(np.dot((Im[Bnet[B1,i,1],:] + Bfc[B1,:] - Bfc[Bnet[B1,i,0]]),lats[fr,:,:]))/np.sqrt(2)
                            templat[B1,i] = temp
                Lati_on[ite,:] = templat
            
            if zi != 2:
                l0 = np.expand_dims(Lati_on[:,:,0], axis=2)
                l1 = np.expand_dims(Lati_on[:,:,1], axis=2)
                l2 = np.expand_dims(Lati_on[:,:,2], axis=2)
                if zi == 0:
                    Lati_on = np.concatenate((l2,l1,l0),axis=2)
                if zi == 1:
                    Lati_on = np.concatenate((l0,l2,l1),axis=2)
            
            Lati_off = np.zeros((len(range(round(traj.nframe*allow_equil),traj.nframe)),Bfc.shape[0],3))
            
            for ite, fr in enumerate(range(round(traj.nframe*allow_equil),traj.nframe)):
                Bfc = Bfrac[fr,:,:]
                templat = np.empty((Bfc.shape[0],3))
                for B1 in range(Bfc.shape[0]):
                    for i in range(3):
                        if i == 2: # identified z-axis
                            templat[B1,i] = np.linalg.norm(np.dot((Im[Bnet[B1,i,1],:] + Bfc[B1,:] - Bfc[Bnet[B1,i,0]]),lats[fr,:,:]))/2
                        else: # x- and y-axis
                            temp = np.linalg.norm(np.dot((Im[Bnet[B1,i,1],:] + Bfc[B1,:] - Bfc[Bnet[B1,i,0]]),lats[fr,:,:]))/np.sqrt(2)
                            if code[B1] == 1:
                                i = (i+1)%2
                            templat[B1,i] = temp
                Lati_off[ite,:] = templat
            
            if zi != 2:
                l0 = np.expand_dims(Lati_off[:,:,0], axis=2)
                l1 = np.expand_dims(Lati_off[:,:,1], axis=2)
                l2 = np.expand_dims(Lati_off[:,:,2], axis=2)
                if zi == 0:
                    Lati_off = np.concatenate((l2,l1,l0),axis=2)
                if zi == 1:
                    Lati_off = np.concatenate((l0,l2,l1),axis=2)
                    
            std_off = norm.fit(Lati_off.reshape(-1,3)[:,0])[1]+norm.fit(Lati_off.reshape(-1,3)[:,1])[1]
            std_on = norm.fit(Lati_on.reshape(-1,3)[:,0])[1]+norm.fit(Lati_on.reshape(-1,3)[:,1])[1]
            
            if std_off > std_on:
                return Lati_on
            else:
                return Lati_off
        
        else: # cells larger than 222 supercell
            
            #Bcart = st0.cart_coords[blist,:]
            Bnet = np.zeros((len(blist),3))
            
            for B1 in range(len(blist)):
                for v in range(3):
                    pos1 = bin_indices[:,B1]+maps_frac[v,:]
                    pos1[pos1>tss] = pos1[pos1>tss]-tss
                    pos1[pos1<1] = pos1[pos1<1]+tss
                    Bnet[B1,v] = np.where(~(bin_indices-pos1[:,np.newaxis]).any(axis=0))[0][0]
            Bnet = Bnet.astype(int)
            
            # find the correct a and b vectors
            #code = np.remainder(np.sum(grided,axis=1),2)
            code = np.remainder(grided[:,1],2)
            trajnum = list(range(round(traj.nframe*allow_equil),traj.nframe,traj.read_every))
            lats = traj.latmat[trajnum,:]
            Bpos = traj.Allpos[trajnum,:,:][:,traj.Bindex,:]
            
            # filtering direction 1
            Lati_off = np.empty((len(trajnum),len(blist),3))
            for B1 in range(len(blist)):
                for i in range(3):
                    if i == 2: # identified z-axis
                        Lati_off[:,B1,i] = (np.linalg.norm(apply_pbc_cart_vecs((Bpos[:,B1,:]-Bpos[:,Bnet[B1,i],:])[:,np.newaxis,:],lats),axis=2)/2).reshape(-1,)
                    else: # x- and y-axis
                        temp = (np.linalg.norm(apply_pbc_cart_vecs((Bpos[:,B1,:]-Bpos[:,Bnet[B1,i],:])[:,np.newaxis,:],lats),axis=2)/np.sqrt(2)).reshape(-1,)
                        if code[B1] == 1:
                            i = (i+1)%2
                        Lati_off[:,B1,i] = temp

            
            if zi != 2:
                l0 = np.expand_dims(Lati_off[:,:,0], axis=2)
                l1 = np.expand_dims(Lati_off[:,:,1], axis=2)
                l2 = np.expand_dims(Lati_off[:,:,2], axis=2)
                if zi == 0:
                    Lati_off = np.concatenate((l2,l1,l0),axis=2)
                if zi == 1:
                    Lati_off = np.concatenate((l0,l2,l1),axis=2)
            
            # filtering direction 2
            Lati_on = np.empty((len(trajnum),len(blist),3))
            for B1 in range(len(blist)):
                for i in range(3):
                    if i == 2: # identified z-axis
                        Lati_on[:,B1,i] = (np.linalg.norm(apply_pbc_cart_vecs((Bpos[:,B1,:]-Bpos[:,Bnet[B1,i],:])[:,np.newaxis,:],lats),axis=2)/2).reshape(-1,)
                    else: # x- and y-axis
                        temp = (np.linalg.norm(apply_pbc_cart_vecs((Bpos[:,B1,:]-Bpos[:,Bnet[B1,i],:])[:,np.newaxis,:],lats),axis=2)/np.sqrt(2)).reshape(-1,)
                        Lati_on[:,B1,i] = temp
            
            if zi != 2:
                l0 = np.expand_dims(Lati_on[:,:,0], axis=2)
                l1 = np.expand_dims(Lati_on[:,:,1], axis=2)
                l2 = np.expand_dims(Lati_on[:,:,2], axis=2)
                if zi == 0:
                    Lati_on = np.concatenate((l2,l1,l0),axis=2)
                if zi == 1:
                    Lati_on = np.concatenate((l0,l2,l1),axis=2)
            
            std_off = norm.fit(Lati_off.reshape(-1,3)[:,0])[1]+norm.fit(Lati_off.reshape(-1,3)[:,1])[1]
            std_on = norm.fit(Lati_on.reshape(-1,3)[:,0])[1]+norm.fit(Lati_on.reshape(-1,3)[:,1])[1]
            
            if std_off > std_on:
                return Lati_on
            else:
                return Lati_off
    
    # begin of main function
    st0 = traj.st0
    Bpos = st0.cart_coords[traj.Bindex,:]
    blist = traj.Bindex
    if len(blist) == 0:
        raise TypeError("Detected zero B site atoms, check B_list")

    #dm = distance_matrix_handler(Bpos,Bpos,st0.lattice.matrix,traj.at0.cell,traj.at0.pbc,traj.complex_pbc)
    #dm = dm.reshape((dm.shape[0]**2,1))
    BBdist = traj.default_BB_dist #np.mean(dm[np.logical_and(dm>0.1,dm<7)])
    
    #if not traj._non_orthogonal:
    celldim = np.cbrt(len(traj.Bindex))
    if not celldim.is_integer():
        raise ValueError("The cell is not in cubic shape, please use lat_method = 1. ")
    
    tss = traj.supercell_size
    cc = st0.frac_coords[blist]
    
    for i in range(3):
        if np.amax(cc[:,i])>(1-1/tss/4) and np.amin(cc[:,i])<1/tss/4:
            addit = np.zeros((1,3))
            addit[0,i] = 1/tss/2
            cc = cc+addit
    cc[cc>1] = cc[cc>1]-1
    cc[cc<0] = cc[cc<0]+1
    
    clims = np.array([[(np.quantile(cc[:,0],1/(tss**2))+np.amin(cc[:,0]))/2,(np.quantile(cc[:,0],1-1/(tss**2))+np.amax(cc[:,0]))/2],
                      [(np.quantile(cc[:,1],1/(tss**2))+np.amin(cc[:,1]))/2,(np.quantile(cc[:,1],1-1/(tss**2))+np.amax(cc[:,1]))/2],
                      [(np.quantile(cc[:,2],1/(tss**2))+np.amin(cc[:,2]))/2,(np.quantile(cc[:,2],1-1/(tss**2))+np.amax(cc[:,2]))/2]])
    
    bin_indices = binstat(cc, None, 'count', bins=[tss,tss,tss], 
                          range=[[clims[0,0]-0.5*(1/tss), 
                                  clims[0,1]+0.5*(1/tss)], 
                                 [clims[1,0]-0.5*(1/tss), 
                                  clims[1,1]+0.5*(1/tss)],
                                 [clims[2,0]-0.5*(1/tss), 
                                  clims[2,1]+0.5*(1/tss)]],
                          expand_binnumbers=True).binnumber
    # validate the binning
    atom_indices = np.array([bin_indices[0,i]+(bin_indices[1,i]-1)*tss+(bin_indices[2,i]-1)*tss**2 for i in range(bin_indices.shape[1])])
    bincount = np.unique(atom_indices, return_counts=True)[1]
    if len(bincount) != tss**3:
        raise TypeError("Incorrect number of bins. ")
    if max(bincount) != min(bincount):
        raise ValueError("Not all bins contain exactly the same number of atoms (1). ")    

    if zdrc == -1:
        Lat = []
        for zi in [0,1,2]:
            Lat.append(run_pcl(zi))
        return Lat
    else:
        return run_pcl(zdrc)         


def structure_time_average_ase(traj, start_ratio = 0.5, end_ratio = 0.98, cif_save_path = None, force_periodicity = False):
    """ 
    Compute the time-averaged structure through the ASE interface. 

    Args:
        traj (Trajectory): The main class instance.
        start_ratio (float): The starting ratio of the trajectory to be considered.
        end_ratio (float): The ending ratio of the trajectory to be considered.
        cif_save_path (str): The path to save the CIF file.
        force_periodicity (bool): If the periodic images are considered.
        
    Returns:
        pymatgen.Structure: The time-averaged Pymatgen structure object
    """
    import pymatgen.io.ase
    # current problem: can't average lattice parameters and angles; can't deal with organic A-site
    
    if force_periodicity:
        lat_param = np.mean(traj.latmat[round(traj.nframe*start_ratio):round(traj.nframe*end_ratio),:],axis=0)
        ap = traj.Allpos[round(traj.nframe*start_ratio):round(traj.nframe*end_ratio),:]
        
        dispp = ap-ap[-1,:,:]
        dispp = apply_pbc_cart_vecs(dispp,traj.latmat[round(traj.nframe*start_ratio):round(traj.nframe*end_ratio),:])
        
        carts = np.mean(dispp,axis=0)+ap[-1,:,:]
        
    else:
        frac = get_frac_from_cart(traj.Allpos, traj.latmat)
        lat_param = np.mean(traj.latmat[round(traj.nframe*start_ratio):round(traj.nframe*end_ratio),:],axis=0)
        CC = np.mean(frac[round(traj.nframe*start_ratio):round(traj.nframe*end_ratio),:],axis=0)
        carts = np.dot(CC,lat_param)


    at=pymatgen.io.ase.AseAtomsAdaptor.get_atoms(traj.st0)
    at.positions = carts
    
    at.set_cell(lat_param)
    struct = pymatgen.io.ase.AseAtomsAdaptor.get_structure(at)
    if cif_save_path:
        struct.to(filename = cif_save_path,fmt="cif")
    
    return struct


def structure_time_average_ase_organic(traj, tavgspan, start_ratio = 0.5, end_ratio = 0.98, cif_save_path = None):
    """ 
    Compute the time-averaged structure, with the organic A-site treated differently. 

    Args:
        traj (Trajectory): The main class instance.
        tavgspan (int): The time span for averaging.
        start_ratio (float): The starting ratio of the trajectory to be considered.
        end_ratio (float): The ending ratio of the trajectory to be considered.
        cif_save_path (str): The path to save the CIF file.

    Returns:
        pymatgen.Structure: The time-averaged Pymatgen structure object
    """
    import pymatgen.io.ase
    # current problem: can't average lattice parameters and angles; can't deal with organic A-site
    
    frac = get_frac_from_cart(traj.Allpos, traj.latmat)
    org_index = traj.Cindex+traj.Hindex+traj.Nindex
    
    endind = min(round(traj.nframe*end_ratio),round(traj.nframe*start_ratio)+tavgspan)
    
    # time averaging
    lat_param = np.mean(traj.latmat[round(traj.nframe*start_ratio):round(traj.nframe*end_ratio),:],axis=0)
    CC = np.mean(frac[round(traj.nframe*start_ratio):round(traj.nframe*end_ratio),:],axis=0)
    orgs = np.mean(frac[round(traj.nframe*start_ratio):endind,org_index,:],axis=0)
    CC[org_index,:] = orgs

    carts = np.dot(CC,lat_param)

    at=pymatgen.io.ase.AseAtomsAdaptor.get_atoms(traj.st0)
    at.positions = carts
    
    at.set_cell(lat_param)
    struct = pymatgen.io.ase.AseAtomsAdaptor.get_structure(at)
    if cif_save_path:
        struct.to(filename = cif_save_path,fmt="cif")
    
    return struct


def simply_calc_distortion(traj):
    """ 
    Compute the octahedral distortion of the time-averaged structure. 

    Args:
        traj (Trajectory): The main class instance.
        
    Returns:
        numpy.ndarray: The octahedral distortion.
        numpy.ndarray: The standard deviations of octahedral distortion.    
    """
    
    struct = traj.tavg_struct
    neigh_list = traj.octahedra
    
    assert struct.atomic_numbers == traj.st0.atomic_numbers
    
    Bpos = struct.cart_coords[traj.Bindex,:]
    Xpos = struct.cart_coords[traj.Xindex,:]
    
    mymat = struct.lattice.matrix
    
    disto = np.empty((0,dist_dim))
    Rmat = np.zeros((len(traj.Bindex),3,3))
    Rmsd = np.zeros((len(traj.Bindex),1))
    for B_site in range(len(traj.Bindex)): # for each B-site atom
        if np.isnan(neigh_list[B_site,:]).any(): 
            a = np.empty((1,dist_dim))
            a[:] = np.nan
            Rmat[B_site,:] = np.nan
            Rmsd[B_site] = np.nan
            disto = np.concatenate((disto,a),axis = 0)
        else:
            raw = Xpos[neigh_list[B_site,:].astype(int),:] - Bpos[B_site,:]
            bx = octahedra_coords_into_bond_vectors(raw,mymat)
            dist_val,rotmat,rmsd = calc_distortions_from_bond_vectors_full(bx)
            Rmat[B_site,:] = rotmat
            Rmsd[B_site] = rmsd
            disto = np.concatenate((disto,dist_val.reshape(1,dist_dim)),axis = 0)
    
    temp_var = np.var(disto,axis = 0)
    temp_std = np.divide(np.sqrt(temp_var),np.nanmean(disto, axis=0))
    temp_dist = np.nanmean(disto,axis=0)
    
    # set all values under tol to be zero.
    tol = 1e-15
    for i in range(len(temp_dist)):
        if temp_dist[i]<tol:
            temp_dist[i] = 0
    
    return temp_dist, temp_std


def octahedra_coords_into_bond_vectors(raw,mymat):
    """ 
    Convert the coordinates of an octahedron to six B-X bond vectors. 

    Args:
        raw (numpy.ndarray): The raw coordinates of an octahedron.
        mymat (numpy.ndarray): The lattice matrix of the structure.

    Returns:
        numpy.ndarray: The six B-X bond vectors.
    """
    bx1 = apply_pbc_cart_vecs(raw, mymat) 
    bx = bx1/np.mean(np.linalg.norm(bx1,axis=1))
    return bx


def calc_distortions_from_bond_vectors(bx):
    """ 
    Compute the tilting and distortion from the six B-X bond vectors. 

    Args:
        bx (numpy.ndarray): The six B-X bond vectors.

    Returns:
        numpy.ndarray: The octahedral distortion.
        numpy.ndarray: Rotation matrix of the octahedron with respect to the reference.
        float: The RMSD of the fitting.
    """
    # constants
    ideal_coords = [[-1, 0,  0],
                    [0, -1,  0],
                    [0,  0, -1],
                    [0,  0,  1],
                    [0,  1,  0],
                    [1,  0,  0]]

    irrep_distortions = []
    for irrep in dict_basis.keys():
        for elem in dict_basis[irrep]:
            irrep_distortions.append(elem)

    # define "molecules"
    pymatgen_molecule = Molecule(
        species=[Element("Pb"),Element("H"),Element("He"),Element("Li"),Element("Be"),Element("B"),Element("I")],
        coords=np.concatenate((np.zeros((1, 3)), bx), axis=0))

    pymatgen_molecule_ideal = Molecule(
        species=pymatgen_molecule.species,
        coords=np.concatenate((np.zeros((1, 3)), ideal_coords), axis=0))

    # transform
    pymatgen_molecule,rotmat,rmsd = match_molecules_extra(
        pymatgen_molecule, pymatgen_molecule_ideal)

    # project
    distortion_amplitudes = calc_displacement(
        pymatgen_molecule, pymatgen_molecule_ideal, irrep_distortions
    )

    # average
    distortion_amplitudes = distortion_amplitudes * distortion_amplitudes
    temp_list = []
    count = 0
    for irrep in dict_basis:
        dim = len(dict_basis[irrep])
        temp_list.append(np.sum(distortion_amplitudes[count:count + dim]))
        count += dim
    distortion_amplitudes = np.sqrt(temp_list)[3:]

    return distortion_amplitudes,rotmat,rmsd


def calc_distortions_from_bond_vectors_full(bx):
    """ 
    Compute the tilting and distortion from the six B-X bond vectors. 

    Args:
        bx (numpy.ndarray): The six B-X bond vectors.

    Returns:
        numpy.ndarray: The octahedral distortion.
        numpy.ndarray: Rotation matrix of the octahedron with respect to the reference.
        float: The RMSD of the fitting.
    """
    # constants
    ideal_coords = [[-1, 0,  0],
                    [0, -1,  0],
                    [0,  0, -1],
                    [0,  0,  1],
                    [0,  1,  0],
                    [1,  0,  0]]

    irrep_distortions = []
    for irrep in dict_basis.keys():
        for elem in dict_basis[irrep]:
            irrep_distortions.append(elem)

    # define "molecules"
    pymatgen_molecule = Molecule(
        species=[Element("Pb"),Element("H"),Element("He"),Element("Li"),Element("Be"),Element("B"),Element("I")],
        coords=np.concatenate((np.zeros((1, 3)), bx), axis=0))

    pymatgen_molecule_ideal = Molecule(
        species=pymatgen_molecule.species,
        coords=np.concatenate((np.zeros((1, 3)), ideal_coords), axis=0))

    # transform
    pymatgen_molecule,rotmat,rmsd = match_molecules_extra(
        pymatgen_molecule, pymatgen_molecule_ideal)

    # project
    distortion_amplitudes = calc_displacement_full(
        pymatgen_molecule, pymatgen_molecule_ideal, irrep_distortions
    )

    # average
    distortion_amplitudes = distortion_amplitudes * distortion_amplitudes
    temp_list = []
    count = 0
    for irrep in dict_basis:
        dim = len(dict_basis[irrep])
        temp_list.append(np.sum(distortion_amplitudes[count:count + dim]))
        count += dim
    distortion_amplitudes = np.sqrt(temp_list)[3:]

    return distortion_amplitudes,rotmat,rmsd


def calc_rotation_from_arbitrary_order(bx):
    """ 
    Compute the rotation of six bond vectors with arbitrary order, used in non-ortho structure. 

    Args:
        bx (numpy.ndarray): The six B-X bond vectors.

    Returns:
        numpy.ndarray: The rotation angles.
        numpy.ndarray: The rotation matrix.
    """
    from scipy.spatial.transform import Rotation as sstr
    
    # constants
    ideal_coords = [[-1, 0,  0],
                    [0, -1,  0],
                    [0,  0, -1],
                    [0,  0,  1],
                    [0,  1,  0],
                    [1,  0,  0]]

    # define "molecules"
    pymatgen_molecule = Molecule(
        species=[Element("Pb"),Element("H"),Element("H"),Element("H"),Element("H"),Element("H"),Element("H")],
        coords=np.concatenate((np.zeros((1, 3)), bx), axis=0))

    pymatgen_molecule_ideal = Molecule(
        species=pymatgen_molecule.species,
        coords=np.concatenate((np.zeros((1, 3)), ideal_coords), axis=0))

    # transform
    pymatgen_molecule,rotmat,rmsd = match_molecules_extra(
        pymatgen_molecule, pymatgen_molecule_ideal)
    
    if rmsd > 0.1:
        raise ValueError(f"The RMSD of fitting related to non-orthogonal frame mapping is too large (RMSD = {round(rmsd,4)})")
    
    rots = sstr.from_matrix(rotmat).as_euler('xyz', degrees=True)
    rots = periodicity_fold(rots)
    
    for i in range(len(rots)):
        if abs(rots[i]) < 0.01:
            rots[i] = 0
            
    rotmat = sstr.from_rotvec(rots/180*np.pi).as_matrix()
            
    return rots, rotmat


def calc_distortion_from_order_manual(bx,ideal_coords,dict_basis):
    """ 
    Compute the rotation of six bond vectors with arbitrary order, used in non-ortho structure. 

    Args:
        bx (numpy.ndarray): The six B-X bond vectors.
        ideal_coords (numpy.ndarray): The ideal coordinates.
        dict_basis (dict): The basis dictionary imported.

    Returns:
        numpy.ndarray: The distortion amplitudes.
        numpy.ndarray: The rotation matrix.
        float: The RMSD of the fitting.
    """
    from scipy.spatial.transform import Rotation as sstr
    
    irrep_distortions = []
    for irrep in dict_basis.keys():
        for elem in dict_basis[irrep]:
            irrep_distortions.append(elem)
    
    # define "molecules"
    pymatgen_molecule = Molecule(
        species=[Element("Pb"),Element("H"),Element("He"),Element("Li"),Element("Be"),Element("B"),Element("I")],
        coords=np.concatenate((np.zeros((1, 3)), bx), axis=0))

    pymatgen_molecule_ideal = Molecule(
        species=pymatgen_molecule.species,
        coords=np.concatenate((np.zeros((1, 3)), ideal_coords), axis=0))

    # transform
    pymatgen_molecule,rotmat,rmsd = match_molecules_extra(
        pymatgen_molecule, pymatgen_molecule_ideal)

    # project
    distortion_amplitudes = calc_displacement_full(
        pymatgen_molecule, pymatgen_molecule_ideal, irrep_distortions
    )

    # average
    distortion_amplitudes = distortion_amplitudes * distortion_amplitudes
    temp_list = []
    count = 0
    for irrep in dict_basis:
        dim = len(dict_basis[irrep])
        temp_list.append(np.sum(distortion_amplitudes[count:count + dim]))
        count += dim
    distortion_amplitudes = np.sqrt(temp_list)

    return distortion_amplitudes,rotmat,rmsd


def quick_match_octahedron(bx):
    """ 
    Compute the rotation status of an octahedron with a reference. 
    Used in connectivity type 3.

    Args:
        bx (numpy.ndarray): The six B-X bond vectors.

    Returns:    
        numpy.ndarray: The rotation matrix.
        numpy.ndarray: The rotated coordinates.
        float: The RMSD of the fitting.
    """
    
    def calc_match(bx):
        # define "molecules"
        pymatgen_molecule = Molecule(
            species=[Element("Pb"),Element("I"),Element("I"),Element("I"),Element("I"),Element("I"),Element("I")],
            coords=np.concatenate((np.zeros((1, 3)), bx), axis=0))

        pymatgen_molecule_ideal = Molecule(
            species=pymatgen_molecule.species,
            coords=np.concatenate((np.zeros((1, 3)), ideal_coords), axis=0))

        # transform
        new_molecule,rotmat,rmsd = match_molecules_extra(pymatgen_molecule_ideal, pymatgen_molecule)
        
        return new_molecule,rotmat,rmsd
        
    # constants
    ideal_coords = [[-1, 0,  0],
                    [0, -1,  0],
                    [0,  0, -1],
                    [0,  0,  1],
                    [0,  1,  0],
                    [1,  0,  0]]

    irrep_distortions = []
    for irrep in dict_basis.keys():
        for elem in dict_basis[irrep]:
            irrep_distortions.append(elem)
 
    new_molecule,rotmat,rmsd = calc_match(bx)
    
    if rmsd > 0.1: # in case of very rare mis-match, shuffle the entry to try again 
        fitting_tol = 0.3
        order = []
        for ix in range(6):
            fits = np.dot(bx,np.array(ideal_coords)[ix,:])
            if not (fits[fits.argsort()[-1]]-fits[fits.argsort()[-2]] > fitting_tol):
                #print(bx,fits)
                raise ValueError(f"The fitting of initial octahedron config to ideal coords is not successful. The initial structure may be too tilted or distorted. The fitting confidence is {round(fits[fits.argsort()[-1]]-fits[fits.argsort()[-2]],3)}, if this is not too far from the tolerance {fitting_tol}, consider lower the fitting_tol in this function. ")
            order.append(np.argmax(fits))
        new_molecule,rotmat,rmsd = calc_match(bx[order,:])
        if rmsd > 0.1: 
            print(bx[order,:])
            #raise RuntimeError("The fitting of initial octahedron config to ideal coords is not successful.")
        
    new_coords = np.matmul(ideal_coords,rotmat)
    #new_coords = new_molecule.cart_coords[1:,:]
    
    return np.linalg.inv(rotmat), new_coords, rmsd


def match_bx_orthogonal(bx):
    """ 
    Find the order of atoms in octahedron through matching with the reference. 
    Used in structure_type 1.

    Args:
        bx (numpy.ndarray): The six B-X bond vectors.

    Returns:    
        list: The order of atoms matching to the reference.
    """
    fitting_tol = 0.3
    ideal_coords = np.array([[-1, 0,  0],
                             [0, -1,  0],
                             [0,  0, -1],
                             [0,  0,  1],
                             [0,  1,  0],
                             [1,  0,  0]])
    order = []
    for ix in range(6):
        fits = np.dot(bx,ideal_coords[ix,:])
        if not (fits[fits.argsort()[-1]]-fits[fits.argsort()[-2]] > fitting_tol):
            #print(bx,fits)
            raise ValueError(f"The fitting of initial octahedron config to ideal coords is not successful. The initial structure may be too tilted or distorted. The fitting confidence is {round(fits[fits.argsort()[-1]]-fits[fits.argsort()[-2]],3)}, if this is not too far from the tolerance {fitting_tol}, consider lower the fitting_tol in this function. ")
        order.append(np.argmax(fits))
    if len(set(order)) != 6: raise ValueError # double-check sanity
    return order


def match_bx_orthogonal_rotated(bx,rotmat):
    """ 
    Find the order of atoms in octahedron through matching with the reference. 
    Used in structure_type 2.

    Args:
        bx (numpy.ndarray): The six B-X bond vectors.
        rotmat (numpy.ndarray): The rotation matrix for alignment.

    Returns:
        list: The order of atoms matching to the reference.
    """
    fitting_tol = 0.3
    ideal_coords = np.array([[-1, 0,  0],
                             [0, -1,  0],
                             [0,  0, -1],
                             [0,  0,  1],
                             [0,  1,  0],
                             [1,  0,  0]])
    ideal_coords = np.matmul(ideal_coords,rotmat)
    order = []
    for ix in range(6):
        fits = np.dot(bx,ideal_coords[ix,:])
        if not (fits[fits.argsort()[-1]]-fits[fits.argsort()[-2]]) > fitting_tol:
            print(bx,fits)
            raise ValueError(f"The fitting of initial octahedron config to ideal coords is not successful. The initial structure may be too tilted or distorted. The fitting confidence is {round(fits[fits.argsort()[-1]]-fits[fits.argsort()[-2]],3)}, if this is not too far from the tolerance {fitting_tol}, consider lower the fitting_tol in this function. ")
        order.append(np.argmax(fits))
    assert len(set(order)) == 6 # double-check sanity
    return order


def match_bx_arbitrary(bx):
    """ 
    Find the order of atoms in octahedron through matching with the reference. 
    Used in structure_type 3.

    Args:
        bx (numpy.ndarray): The six B-X bond vectors.

    Returns:
        list: The order of atoms matching to the reference.
        numpy.ndarray: rotation matrices for alignment.
        float: The RMSD of the fitting.
    """
    fitting_tol = 0.4
    #confidence_bound = [0.8,0.2]
    ref_rot, ideal_coords, rmsd = quick_match_octahedron(bx)
    
    order = []
    for ix in range(6):
        fits = np.dot(bx,ideal_coords[ix,:])
        #if not (fits[fits.argsort()[-1]] > confidence_bound[0] and fits[fits.argsort()[-2]] < confidence_bound[1]):
        if not (fits[fits.argsort()[-1]]-fits[fits.argsort()[-2]]) > fitting_tol:
            print(bx,fits)
            raise ValueError("The fitting of initial octahedron config to rotated reference is not successful. This may happen if the initial configuration is too distorted. ")
        order.append(np.argmax(fits))
    assert len(set(order)) == 6 # double-check sanity
    return order, ref_rot, rmsd


def match_molecules_extra(molecule_transform, molecule_reference):
    """ 
    Hungarian matching to output the transformation matrix

    Args:
        molecule_transform (pymatgen.Molecule): The molecule to be transformed.
        molecule_reference (pymatgen.Molecule): The reference molecule.

    Returns:    
        pymatgen.Molecule: The transformed molecule.
        numpy.ndarray: The rotation matrix.
        float: The RMSD of the fitting.
    """
    
    # match molecules
    (inds, u, v, rmsd) = HungarianOrderMatcher(
        molecule_reference).match(molecule_transform)

    # affine transform
    molecule_transform.apply_operation(SymmOp(
        np.concatenate((
            np.concatenate((u.T, v.reshape(3, 1)), axis=1),
            [np.zeros(4)]), axis=0
        )))
    molecule_transform._sites = np.array(
        molecule_transform._sites)[inds].tolist()

    return molecule_transform,u,rmsd


def calc_displacement(pymatgen_molecule, pymatgen_molecule_ideal, irrep_distortions):
    """
    Compute the displacement of the atoms in the molecule for matching.
    
    Args:
        pymatgen_molecule (pymatgen.Molecule): The molecule to be transformed.
        pymatgen_molecule_ideal (pymatgen.Molecule): The reference molecule.
        irrep_distortions (numpy.ndarray): The basis irreps for matching.

    Returns:    
        numpy.ndarray: The processed displacement of the atoms.
    """
    
    return np.tensordot(irrep_distortions,
                        (pymatgen_molecule.cart_coords - pymatgen_molecule_ideal.cart_coords).ravel()[3:],
                        axes=1)

def calc_displacement_full(pymatgen_molecule, pymatgen_molecule_ideal, irrep_distortions):
    """
    Compute the displacement of the atoms in the molecule for matching.

    Args:
        pymatgen_molecule (pymatgen.Molecule): The molecule to be transformed.
        pymatgen_molecule_ideal (pymatgen.Molecule): The reference molecule.
        irrep_distortions (numpy.ndarray): The basis irreps for matching.

    Returns:
        numpy.ndarray: The processed displacement of the atoms.
    """
    
    return np.tensordot(irrep_distortions,
                        (pymatgen_molecule.cart_coords - pymatgen_molecule_ideal.cart_coords).ravel(),
                        axes=1)


def match_mixed_halide_octa_dot(bx,hals):
    """ 
    Find the configuration class in binary-mixed halide octahedron.

    Args:
        bx (numpy.ndarray): The six B-X bond vectors.
        hals (list): The list of symbols of halide species.

    Returns:
        tuple: The configuration class and the configuration
    """
    
    distinct_thresh = 20
    
    ideal_coords = [
        [-1,  0,  0],
        [0, -1,  0],
        [0,  0, -1],
        [0,  0,  1],
        [0,  1,  0],
        [1,  0,  0]]
    
    brnum = hals.count('Br')
    
    if brnum == 0:
        return (0,0),0
    
    elif brnum == 1:
        return (1,0),1
    
    elif brnum == 2:
        brs = []
        for i, h in enumerate(hals):
            if h=="Br":
                brs.append(i)

        ang = np.dot(bx[brs[0],:],bx[brs[1],:])
        
        if ang < -0.7: # planar form
            return (2,1),3 # planar form
        elif abs(ang) < 0.3: # right-angle form
            return (2,0),2 # right-angle form
        else:
            print(ang)
            raise ValueError("Can't distinguish the local halide environment. Angular term printed above. ")
    
    elif brnum == 3:
        brs = []
        for i, h in enumerate(hals):
            if h=="Br":
                brs.append(i)

        ang = [np.dot(bx[brs[0],:],bx[brs[1],:]),np.dot(bx[brs[0],:],bx[brs[2],:]),np.dot(bx[brs[1],:],bx[brs[2],:])]
        
        if min(ang) < -0.7: # planar form
            return (3,1),5 # planar form
        elif abs(min(ang)) < 0.3: # right-angle form
            return (3,0),4 # right-angle form
        else:
            print(ang)
            raise ValueError("Can't distinguish the local halide environment. Angular terms printed above.")
    
    elif brnum == 4:
        ios = []
        for i, h in enumerate(hals):
            if h=="I":
                ios.append(i)

        ang = np.dot(bx[ios[0],:],bx[ios[1],:])
        
        if ang < -0.7: # planar form
            return (4,1),7 # planar form
        elif abs(ang) < 0.3: # right-angle form
            return (4,0),6 # right-angle form
        else:
            print(ang)
            raise ValueError("Can't distinguish the local halide environment. Angular term printed above. ")

    elif brnum == 5:
        return (5,0),8

    elif brnum == 6:
        return (6,0),9


def centmass_organic(st0pos,latmat,env):
    """ 
    Find the center of mass of organic A-site.

    Args:
        st0pos (numpy.ndarray): The atomic positions.
        latmat (numpy.ndarray): The lattice matrix.
        env (list): The environment of the A-site.

    Returns:
        numpy.ndarray: The center of mass positions.
    """
    
    c = env[0]
    n = env[1]
    h = env[2]
    
    refc = st0pos[[c[0]],:]
    cs = apply_pbc_cart_vecs(st0pos[c,:] - refc,latmat)
    ns = apply_pbc_cart_vecs(st0pos[n,:] - refc,latmat)
    hs = apply_pbc_cart_vecs(st0pos[h,:] - refc,latmat)
    
    mass = refc+(np.sum(cs,axis=0)*12+np.sum(ns,axis=0)*14+np.sum(hs,axis=0)*0)/(12*len(c)+14*len(n)+1*len(h))

    return mass


def centmass_organic_vec(pos,latmat,env):
    """ 
    Find the center of mass of organic A-site. (vectorised version)

    Args:
        pos (numpy.ndarray): The atomic positions.
        latmat (numpy.ndarray): The lattice matrix.
        env (list): The environment of the A-site.

    Returns:
        numpy.ndarray: The center of mass positions.
    """
    
    c = env[0]
    n = env[1]
    h = env[2]
    
    refc = pos[:,[c[0]],:]
    cs = apply_pbc_cart_vecs(pos[:,c,:] - refc,latmat)
    ns = apply_pbc_cart_vecs(pos[:,n,:] - refc,latmat)
    hs = apply_pbc_cart_vecs(pos[:,h,:] - refc,latmat)
    
    mass = refc[:,0,:]+(np.sum(cs,axis=1)*12+np.sum(ns,axis=1)*14+np.sum(hs,axis=1)*1)/(12*len(c)+14*len(n)+1*len(h))

    return mass


def find_B_cage_and_disp(pos,mymat,cent,Bs):
    """ 
    Find the displacement of A-site with respect to the capsulating B-X cage.

    Args:
        pos (numpy.ndarray): The atomic positions.
        mymat (numpy.ndarray): The lattice matrix.
        cent (numpy.ndarray): The center of mass of A-site.
        Bs (list): The indices of the B-site atoms.

    Returns:
        numpy.ndarray: The relative displacement of A-site to its surrounding.
    """
    
    B8pos = pos[:,Bs,:]
    BX = B8pos - np.expand_dims(cent, axis=1)
    
    BX = apply_pbc_cart_vecs(BX, mymat)
    
    Bs = BX + np.expand_dims(cent, axis=1)
    Xcent = np.mean(Bs, axis=1)
    
    disp = cent - Xcent
    
    return disp
    #return np.expand_dims(disp, axis=1)


def periodicity_fold(arrin,n_fold=4):
    """ 
    Handle abnormal tilting numbers. 

    Args:
        arrin (numpy.ndarray): The input array of tilting angles.
        n_fold (int): nfold of angle range.

    Returns:
        numpy.ndarray: The folded tilting angles.
    """
    from copy import deepcopy
    arr = deepcopy(arrin)
    if n_fold == 4:
        arr[arr<-45] = arr[arr<-45]+90
        arr[arr<-45] = arr[arr<-45]+90
        arr[arr>45] = arr[arr>45]-90
        arr[arr>45] = arr[arr>45]-90
    elif n_fold == 2:
        arr[arr<-90] = arr[arr<-90]+180
        arr[arr>90] = arr[arr>90]-180
    elif n_fold == 8:
        arr[arr<-45] = arr[arr<-45]+90
        arr[arr<-45] = arr[arr<-45]+90
        arr[arr>45] = arr[arr>45]-90
        arr[arr>45] = arr[arr>45]-90
        arr = np.abs(arr)
    return arr


def get_cart_from_frac(frac,latmat):
    """ 
    Get cartesian coordinates from fractional coordinates. 

    Args:
        frac (numpy.ndarray): The fractional coordinates.
        latmat (numpy.ndarray): The lattice matrix.

    Returns:
        numpy.ndarray: The corresponding cartesian coordinates.
    """
    if frac.ndim != latmat.ndim:
        raise ValueError("The dimension of the input arrays do not match. ")
    if frac.shape[-1] != 3 or latmat.shape[-2:] != (3,3):
        raise TypeError("Must be 3D vectors. ")
        
    if frac.ndim == 2:
        pass
    elif frac.ndim == 3:
        if frac.shape[0] != latmat.shape[0]:
            raise ValueError("The frame number of input arrays do not match. ")
    else:
        raise TypeError("Can only deal with 2 or 3D arrays. ")
            
    return np.matmul(frac,latmat)


def get_frac_from_cart(cart,latmat):
    """ 
    Get fractional coordinates from cartesian coordinates. 

    Args:
        cart (numpy.ndarray): The cartesian coordinates.
        latmat (numpy.ndarray): The lattice matrix.
                
    Returns:
        numpy.ndarray: The corresponding fractional coordinates.
    """
    if cart.ndim != latmat.ndim:
        raise ValueError("The dimension of the input arrays do not match. ")
    if cart.shape[-1] != 3 or latmat.shape[-2:] != (3,3):
        raise TypeError("Must be 3D vectors. ")
        
    if cart.ndim == 2:
        pass
    elif cart.ndim == 3:
        if cart.shape[0] != latmat.shape[0]:
            raise ValueError("The frame number of input arrays do not match. ")
    else:
        raise TypeError("Can only deal with 2 or 3D arrays. ")
            
    return np.matmul(cart,np.linalg.inv(latmat))


def apply_pbc_cart_vecs(vecs, mymat):
    """ 
    Apply PBC to a set of vectors wrt. lattice matrix. (vectorised version)

    Args:
        vecs (numpy.ndarray): The vectors.
        mymat (numpy.ndarray): The lattice matrix.

    Returns:
        numpy.ndarray: The corresponding vectors with PBC applied.
    """
    vecs_frac = get_frac_from_cart(vecs, mymat)
    vecs_pbc = get_cart_from_frac(vecs_frac-np.round(vecs_frac), mymat)
    return vecs_pbc

def apply_pbc_cart_vecs_single_frame(vecs, mymat):
    """ 
    Apply PBC to a set of vectors wrt. lattice matrix.  

    Args:   
        vecs (numpy.ndarray): The vectors.
        mymat (numpy.ndarray): The lattice matrix.

    Returns:
        numpy.ndarray: The corresponding vectors with PBC applied.
    """
    vecs_frac = np.matmul(vecs,np.linalg.inv(mymat))
    vecs_pbc = np.matmul(vecs_frac-np.round(vecs_frac), mymat)
    return vecs_pbc


def find_population_gap(r,find_range,init,tol=0):
    """ 
    Find the 1D classifier value to separate two populations. 

    Args:
        r (numpy.ndarray): The input distance matrix array. 
        find_range (list): The distance range to find the classifier. 
        init (numpy.ndarray): The initial guess of the classifier. 
        tol (float): The tolerance of the classifier. 

    Returns:    
        float: The classifier value.
    """
    from scipy.cluster.vq import kmeans
    
    scan = r.reshape(-1,)
    scan = scan[np.logical_and(scan<find_range[1],scan>find_range[0])]
    centers = kmeans(scan, k_or_guess=init, iter=20, thresh=1e-05)[0]
    p = np.mean(centers) 

    y,binEdges=np.histogram(scan,bins=50)
    bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
    
    if y[(np.abs(bincenters - p)).argmin()] > tol:
        p1 = centers[0]+(centers[1]-centers[0])/(centers[1]+centers[0])*centers[0]
        p = p1
        if y[(np.abs(bincenters - p)).argmin()] > tol:
            plt.hist(r.reshape(-1,),bins=100,range=[1,10])
            raise ValueError("!Structure resolving: Can't separate the different neighbours, please check the fpg_val values are correct according to the guidance at the beginning of the Trajectory class, or check if initial structure is defected or too distorted. ")

    return p


def get_volume(lattice):

    """ 
    Calculate the volume of the lattice.

    Args:
        lattice (numpy.ndarray): The lattice matrix.

    Returns:
        float: The volume of the lattice.   
    """

    
    if lattice.shape == (6,):
        A,B,C,a,b,c = lattice
        a=a/180*np.pi
        b=b/180*np.pi
        c=c/180*np.pi
        Vol = A*B*C*(1-np.cos(a)**2-np.cos(b)**2-np.cos(c)**2+2*np.cos(a)*np.cos(b)*np.cos(c))**0.5
    elif (lattice.ndim == 2 and lattice.shape[1] == 6):
        Vol = []
        for i in range(lattice.shape[0]):
            A,B,C,a,b,c = lattice[i,:]
            a=a/180*np.pi
            b=b/180*np.pi
            c=c/180*np.pi
            vol = A*B*C*(1-np.cos(a)**2-np.cos(b)**2-np.cos(c)**2+2*np.cos(a)*np.cos(b)*np.cos(c))**0.5
            Vol.append(vol)
        Vol = np.array(Vol)  
    else:
        raise TypeError("The input must with shape (6,) or (N,6). ")
        
    return Vol


def distance_matrix(v1,v2,latmat,get_vec=False):
    """
    Compute the distance matrix between two sets of vectors, using the apparent matrix multiplication method. 
    Should only be used for the case with alpha, beta, gamma angles all close to 90 degrees.

    Args:
        v1 (numpy.ndarray): The first set of coordinate vectors.
        v2 (numpy.ndarray): The second set of coordinate vectors.
        latmat (numpy.ndarray): The lattice matrix.
        get_vec (bool): Whether to return the distance vectors.

    Returns:
        numpy.ndarray: The distance matrix.
    """
    
    dprec = np.float32
    
    f1 = get_frac_from_cart(v1,latmat)[:,np.newaxis,:].astype(dprec)
    f2 = get_frac_from_cart(v2,latmat)[np.newaxis,:,:].astype(dprec)
    
    df = np.repeat(f1,f2.shape[1],axis=1)-np.repeat(f2,f1.shape[0],axis=0)
    df = df-np.round(df)
    df = np.matmul(df,latmat.astype(dprec))
    
    if get_vec:
        return df
    else:
        return np.linalg.norm(df,axis=2)
  
    
def distance_matrix_ase(v1,v2,asecell,pbc,get_vec=False):   
    """
    Compute the distance matrix between two sets of vectors, using the algorithm implemented in ASE.

    Args:
        v1 (numpy.ndarray): The first set of coordinate vectors.
        v2 (numpy.ndarray): The second set of coordinate vectors.
        asecell (ase.Atoms): The Atoms object of the system.
        pbc (list of bool): Periodic boudary condition in three directions as defined in ASE.
        get_vec (bool): Whether to return the distance vectors.

    Returns:
        numpy.ndarray: The distance matrix.
    """

    from ase.geometry import get_distances
    
    D, D_len = get_distances(v1,v2,cell=asecell, pbc=pbc)
    
    if get_vec:
        return D
    else:
        return D_len
    
    
def distance_matrix_ase_replace(v1,v2,asecell,newcell,pbc,get_vec=False):
    """
    Compute the distance matrix between two sets of vectors, using the algorithm implemented in ASE, with a new cell.
    Uses not the cell object in the Atoms object, but a new cell matrix.

    Args:
        v1 (numpy.ndarray): The first set of coordinate vectors.
        v2 (numpy.ndarray): The second set of coordinate vectors.
        asecell (ase.Atoms): The Atoms object of the system.
        newcell (numpy.ndarray): The new cell matrix.
        pbc (list of bool): Periodic boudary condition in three directions as defined in ASE.
        get_vec (bool): Whether to return the distance vectors.

    Returns:
        numpy.ndarray: The distance matrix.
    """

    from ase.geometry import get_distances
    
    celltemp = asecell.copy()
    celltemp.array = newcell
    
    D, D_len = get_distances(v1,v2,cell=celltemp, pbc=pbc)
    
    if get_vec:
        return D
    else:
        return D_len
    

def distance_matrix_handler(v1,v2,latmat,asecell=None,pbc=None,complex_pbc=False,replace=True,get_vec=False):
    """
    Compute the distance matrix between two sets of vectors, with different methods depending on requirement.

    Args:
        v1 (numpy.ndarray): The first set of coordinate vectors.
        v2 (numpy.ndarray): The second set of coordinate vectors.
        latmat (numpy.ndarray): The lattice matrix.
        asecell (ase.Atoms): The Atoms object of the system.
        pbc (list of bool): Periodic boudary condition in three directions as defined in ASE.
        complex_pbc (bool): Whether to use the ASE algorithm.
        replace (bool): Whether to use a new cell matrix.
        get_vec (bool): Whether to return the distance vectors.

    Returns:
        numpy.ndarray: The distance matrix.
    """
    
    if v1.shape == (3,):
        v1 = v1[np.newaxis,:]
    if v2.shape == (3,):
        v2 = v2[np.newaxis,:]
    if v1.ndim != 2 or v1.shape[1] != 3 or v2.ndim != 2 or v2.shape[1] != 3:
        raise TypeError("The input arrays must be in shape (N*3) or (3,). ")
    
    if complex_pbc is False:
        r = distance_matrix(v1,v2,latmat,get_vec=get_vec)
    else:
        if replace is False:
            r = distance_matrix_ase(v1,v2,asecell,pbc,get_vec=get_vec)
        else:
            r = distance_matrix_ase_replace(v1,v2,asecell,latmat,pbc,get_vec=get_vec)
    
    return r
    



