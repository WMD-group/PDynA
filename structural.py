import numpy as np
import re, os
import json
import sys
from joblib import Parallel, delayed
from tqdm import tqdm
import pymatgen.io.cif
import pymatgen.analysis.molecule_matcher
from pymatgen.core.periodic_table import Element
from scipy.spatial.transform import Rotation as sstr


def resolve_octahedra(Bpos,Xpos,readfr,enable_refit,multi_thread,lattice,neigh_list,ref_with_initial_structure,Rmat_ref,orthogonal_frame):
    
    def frame_wise(fr):
        mybox=lattice[fr,:]
        
        #R[i,:] = distance_array(Bpos[i,:],Xpos[i,:],mybox)
        disto = np.empty((0,4))
        Rmat = np.zeros((Bcount,3,3))
        Rmsd = np.zeros((Bcount,1))
        for B_site in range(Bcount): # for each B-site atom
                
            raw = Xpos[fr,neigh_list[B_site,:],:] - Bpos[fr,B_site,:]
            bx = octahedra_coords_into_bond_vectors(raw,mybox)
            if ref_with_initial_structure:
                bx = np.matmul(bx,Rmat_ref[B_site,:])        
            dist_val,rotmat,rmsd = calc_distortions_from_bond_vectors(bx,orthogonal_frame)
            Rmat[B_site,:] = rotmat
            Rmsd[B_site] = rmsd
            disto = np.concatenate((disto,dist_val.reshape(1,4)),axis = 0)
        
        tilts = np.zeros((Bcount,3))
        for i in range(Rmat.shape[0]):
            tilts[i,:] = sstr.from_matrix(Rmat[i,:]).as_euler('xyz', degrees=True)

        #tilts = periodicity_fold(tilts) 
        return (disto.reshape(1,Bcount,4),tilts.reshape(1,Bcount,3),Rmsd)
    

    Bcount = Bpos.shape[1]
    refits = np.empty((0,2))
    
    if multi_thread == 1: # multi-threading disabled and can do refit
        
        Di = np.empty((len(readfr),Bcount,4))
        T = np.empty((len(readfr),Bcount,3))
        
        for subfr in tqdm(range(len(readfr))):
            fr = readfr[subfr]
            mybox=lattice[fr,:]
            
            #R[i,:] = distance_array(Bpos[i,:],Xpos[i,:],mybox)
            disto = np.empty((0,4))
            Rmat = np.zeros((Bcount,3,3))
            Rmsd = np.zeros((Bcount,1))
            for B_site in range(Bcount): # for each B-site atom

                raw = Xpos[fr,neigh_list[B_site,:],:] - Bpos[fr,B_site,:]
                bx = octahedra_coords_into_bond_vectors(raw,mybox)

                if ref_with_initial_structure:
                    bx = np.matmul(bx,Rmat_ref[B_site,:])        
                dist_val,rotmat,rmsd = calc_distortions_from_bond_vectors(bx,orthogonal_frame)
                Rmat[B_site,:] = rotmat
                Rmsd[B_site] = rmsd
                disto = np.concatenate((disto,dist_val.reshape(1,4)),axis = 0)
            
            if enable_refit and fr > 0 and max(np.amax(disto)) > 0.8: # re-fit neigh_list if large distortion value is found
                from MDAnalysis.analysis.distances import distance_array    
                disto_prev = np.mean(disto,axis=0)
                neigh_list_prev = neigh_list
 
                r=distance_array(Bpos[fr,:],Xpos[fr,:],mybox)
                
                max_BX_distance = 4.2
                neigh_list = np.zeros((Bpos.shape[1],6))
                for B_site, X_list in enumerate(r): # for each B-site atom
                    X_idx = [i for i in range(len(X_list)) if X_list[i] < max_BX_distance]
                    if len(X_idx) != 6:
                        raise ValueError(f"The number of X site atoms connected to B site atom no.{B_site} is not 6 but {len(X_idx)}. \n")
                    
                    if orthogonal_frame:
                        bx_raw = Xpos[fr,:][X_idx,:] - Bpos[fr,:][B_site,:]
                        order1 = match_bx(bx_raw,mybox) 
                        neigh_list[B_site,:] = np.array(X_idx)[order1]
                    else:     
                        neigh_list[B_site,:] = np.array(X_idx)
                    
                neigh_list = neigh_list.astype(int)
                
                if neigh_list.shape != (Bcount, 6):
                    raise ValueError(f"refit of neighbour list at frame no.{fr} is unsuccessful, check neigh_list. ")
                
                mybox=lattice[fr,:]
                
                # re-calculate distortion values
                disto = np.empty((0,4))
                Rmat = np.zeros((Bcount,3,3))
                Rmsd = np.zeros((Bcount,1))
                for B_site in range(Bcount): # for each B-site atom
                        
                    raw = Xpos[fr,neigh_list[B_site,:],:] - Bpos[fr,B_site,:]
                    bx = octahedra_coords_into_bond_vectors(raw,mybox)
                    if ref_with_initial_structure:
                        bx = np.matmul(bx,Rmat_ref[B_site,:])
                    dist_val,rotmat,rmsd = calc_distortions_from_bond_vectors(bx,orthogonal_frame)
                    Rmat[B_site,:] = rotmat
                    Rmsd[B_site] = rmsd
                    disto = np.concatenate((disto,dist_val.reshape(1,4)),axis = 0)
                if np.array_equal(np.mean(disto,axis=0),disto_prev) and np.array_equal(neigh_list,neigh_list_prev):
                    refits = np.concatenate((refits,np.array([[fr,0]])),axis=0)
                else:
                    refits = np.concatenate((refits,np.array([[fr,1]])),axis=0)
            
            Di[subfr,:,:] = disto.reshape(1,Bcount,4)
            
            tilts = np.zeros((Bcount,3))
            for i in range(Rmat.shape[0]):
                tilts[i,:] = sstr.from_matrix(Rmat[i,:]).as_euler('xyz', degrees=True) 
            #tilts = periodicity_fold(tilts)
            T[subfr,:,:] = tilts.reshape(1,Bcount,3)
            
            
    elif multi_thread > 1: # multi-threading enabled and cannot refit
        Di = np.empty((len(readfr),Bcount,4))
        T = np.empty((len(readfr),Bcount,3))
        
        results = Parallel(n_jobs=multi_thread)(delayed(frame_wise)(fr) for fr in readfr)
        
        # unpack the result from multi-threading calculation
        assert len(results) == len(readfr)
        for fr, each in enumerate(results):
            Di[fr,:,:] = each[0]
            T[fr,:,:] = each[1]
    
    else:
        raise ValueError("The selected multi-threading count must be a positive integer.")
        
    
    return Di, T, refits


def pseudocubic_lat(traj,  # the main class instance
                    allow_equil = 0, #take the first x fraction of the trajectory as equilibration
                    zdrc = 2,  # zdrc: the z direction for pseudo-cubic lattice paramter reading. a,b,c = 0,1,2
                    lattice_tilt = [0,0,0], # if primary B-B bond is not along orthogonal directions, in degrees
                    orthor_filter = True):  # Apply a filter to Lat values to distinguish orthor phase a/b lattice parameters, disable if see double peak in both a and b axis
    
    #from MDAnalysis.analysis.distances import distance_array

    match_tol = 1.5 # tolerance of the coordinate difference is set to 1.5 angstroms    

    Im = np.empty((0,3))
    for i in range(-1,2): # create images of all surrounding cells
        for j in range(-1,2):
            for k in range(-1,2):
                Im = np.concatenate((Im,np.array([[i,j,k]])),axis=0) 
    
    st0 = traj.st0
    blist = traj.Bindex
    
    if len(blist) == 0:
        raise TypeError("Detected zero B site atoms, check B_list")
            
    celldim = np.cbrt(len(traj.Bindex))
    if not celldim.is_integer():
        raise ValueError("The cell is not in cubic shape. ")

    
    dm = st0.distance_matrix[blist][:,blist]
    dm = dm.reshape((dm.shape[0]**2,1))
    BBdist = np.mean(dm[np.logical_and(dm>0.1,dm<7)]) # default B-B distance
    
    dims = [0,1,2]
    dims.remove(zdrc)
    cartcoords = st0.cart_coords[blist][:,dims]
    grided = np.round((cartcoords-np.min(cartcoords,axis=0))/BBdist)
    counts = np.unique(grided, return_counts=True)[1]
    if max(counts) != 2*celldim**2 or min(counts) != 2*celldim**2:
        print(st0.symbol_set, 'check if intial structure is not read correctly. ')
        raise TypeError("The B site atoms did not fit in the grids. ")

    mappings = BBdist*np.array([[[0,1,1],[0,1,-1],[2,0,0]],
                              [[-1,0,1],[1,0,1],[0,2,0]],
                              [[-1,1,0],[1,1,0],[0,0,2]]]) # setting of pseuso-cubic lattice parameter vectors
    maps = mappings[zdrc,:,:] # select out of x,y,z
    
    if lattice_tilt != [0,0,0]:
        rot = sstr.from_euler('xyz', lattice_tilt, degrees=True)
        rotmat = rot.as_matrix()
        
        maps = np.dot(maps,rotmat)
    
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
    

    Lati = np.zeros((len(range(round(traj.nframe*allow_equil),traj.nframe)),Bfc.shape[0],3))
    
    # find the correct a and b vectors
    #code = np.remainder(np.sum(grided,axis=1),2)
    code = np.remainder(grided[:,1],2)
    
    Bfrac = traj.Allfrac[:,traj.Bindex,:]
    lats = traj.latmat
    for ite, fr in enumerate(range(round(traj.nframe*allow_equil),traj.nframe)):
        Bfc = Bfrac[fr,:,:]
        templat = np.empty((Bfc.shape[0],3))
        for B1 in range(Bfc.shape[0]):
            for i in range(3):
                if i == 2: # identified z-axis
                    templat[B1,i] = np.linalg.norm(np.dot((Im[Bnet[B1,i,1],:] + Bfc[B1,:] - Bfc[Bnet[B1,i,0]]),lats[fr,:,:]))/2
                else: # x- and y-axis
                    temp = np.linalg.norm(np.dot((Im[Bnet[B1,i,1],:] + Bfc[B1,:] - Bfc[Bnet[B1,i,0]]),lats[fr,:,:]))/np.sqrt(2)
                    if orthor_filter:
                        if code[B1] == 1:
                            i = (i+1)%2
                    templat[B1,i] = temp
        Lati[ite,:] = templat
    
    if zdrc != 2:
        l0 = np.expand_dims(Lati[:,:,0], axis=2)
        l1 = np.expand_dims(Lati[:,:,1], axis=2)
        l2 = np.expand_dims(Lati[:,:,2], axis=2)
        if zdrc == 0:
            Lati = np.concatenate((l2,l1,l0),axis=2)
        if zdrc == 1:
            Lati = np.concatenate((l0,l2,l1),axis=2)
    
    return Lati


def structure_time_average(traj, start_ratio = 0.5, end_ratio = 0.98, cif_save_path = None):
    # current problem: can't average lattice parameters and angles; can't deal with organic A-site
    M_len = traj.nframe
    
    # time averaging
    lat_param = np.concatenate((np.mean(traj.lattice[round(M_len*start_ratio):round(M_len*end_ratio),:3],axis=0).reshape(1,-1),np.mean(traj.lattice[round(M_len*start_ratio):round(M_len*end_ratio),3:],axis=0).reshape(1,-1)),axis=0)
    CC = np.mean(traj.Allfrac[round(M_len*start_ratio):round(M_len*end_ratio),:],axis=0)

    # construct a cif file for modification
    pymatgen.io.cif.CifWriter(traj.st0).write_file("temp1.cif")
    with open("temp1.cif") as file:
        lines = file.readlines()
    for i in range(len(lines)): 
        lines[i] = lines[i].lstrip(' ') # remove all leading spaces in lines
        lines[i] = re.sub(' +\n', '\n', lines[i]) # remove all spaces before '\n' in lines 
    os.unlink('temp1.cif')
    
    # change the lattice param and angles
    for i, line in enumerate(lines):
        if line.startswith("_cell_length_a"):
            lines[i] = re.sub(line.split()[1], str(lat_param[0,0]), line)
        if line.startswith("_cell_length_b"):
            lines[i] = re.sub(line.split()[1], str(lat_param[0,1]), line)
        if line.startswith("_cell_length_c"):
            lines[i] = re.sub(line.split()[1], str(lat_param[0,2]), line)
        if line.startswith("_cell_angle_alpha"):
            lines[i] = re.sub(line.split()[1], str(lat_param[1,0]), line)
        if line.startswith("_cell_angle_beta"):
            lines[i] = re.sub(line.split()[1], str(lat_param[1,1]), line)
        if line.startswith("_cell_angle_gamma"):
            lines[i] = re.sub(line.split()[1], str(lat_param[1,2]), line)
            
    
    # change the atomic positions
    for i, line in enumerate(lines):
        if line.startswith("_atom_site_"):
            break
    i_coord = 0
    while line.startswith("_atom_site_"):
        i+=1
        i_coord+=1
        line = lines[i]
        if line.startswith("_atom_site_fract_x"):
            line_coord = i_coord

    Natom = traj.natom
    k = 0
    for j in range(i,i+Natom):
        line = lines[j].split()
        line[line_coord:line_coord+3] = CC[k,:]
        line = [str(x) for x in line]
        lines[j] = " ".join(line)+'\n'
        k+=1
    
    # construct another cif file for saving
    if cif_save_path is None:
        a_file = open('temp2.cif', "w")
        a_file.writelines(lines)
        a_file.close()
    else:
        b_file = open(cif_save_path, "w")
        b_file.writelines(lines)
        b_file.close()
        
        a_file = open('temp2.cif', "w")
        a_file.writelines(lines)
        a_file.close()
        
    cif_obj = pymatgen.io.cif.CifParser('temp2.cif', occupancy_tolerance=5)
    struct = cif_obj.get_structures(primitive=False)[0]

    os.unlink('temp2.cif')   
    
    return struct


def structure_time_average_ase(traj, start_ratio = 0.5, end_ratio = 0.98, cif_save_path = None):
    import pymatgen.io.ase
    # current problem: can't average lattice parameters and angles; can't deal with organic A-site
    
    # time averaging
    lat_param = np.mean(traj.latmat[round(traj.nframe*start_ratio):round(traj.nframe*end_ratio),:],axis=0)
    CC = np.mean(traj.Allfrac[round(traj.nframe*start_ratio):round(traj.nframe*end_ratio),:],axis=0)

    carts = np.dot(CC,lat_param)

    at=pymatgen.io.ase.AseAtomsAdaptor.get_atoms(traj.st0)
    at.positions = carts
    
    at.set_cell(lat_param)
    struct = pymatgen.io.ase.AseAtomsAdaptor.get_structure(at)
    if cif_save_path:
        struct.to(filename = cif_save_path,fmt="cif")
    
    return struct


def simply_calc_distortion(traj):
    
    struct = traj.tavg_struct
    neigh_list = traj.octahedra
    
    assert struct.atomic_numbers == traj.st0.atomic_numbers
    
    Bpos = struct.cart_coords[traj.Bindex,:]
    Xpos = struct.cart_coords[traj.Xindex,:]
    
    mybox=np.array([struct.lattice.abc,struct.lattice.angles]).reshape(6,)
    
    disto = np.empty((0,4))
    Rmat = np.zeros((len(traj.Bindex),3,3))
    Rmsd = np.zeros((len(traj.Bindex),1))
    for B_site in range(len(traj.Bindex)): # for each B-site atom
    
        raw = Xpos[neigh_list[B_site,:],:] - Bpos[B_site,:]
        bx = octahedra_coords_into_bond_vectors(raw,mybox)
        dist_val,rotmat,rmsd = calc_distortions_from_bond_vectors(bx,True)
        Rmat[B_site,:] = rotmat
        Rmsd[B_site] = rmsd
        disto = np.concatenate((disto,dist_val.reshape(1,4)),axis = 0)
    
    temp_var = np.var(disto,axis = 0)
    temp_std = np.divide(np.sqrt(temp_var),np.mean(disto, axis=0))
    temp_dist = np.mean(disto,axis=0)
    
    # set all values under tol to be zero.
    tol = 1e-15
    for i in range(len(temp_dist)):
        if temp_dist[i]<tol:
            temp_dist[i] = 0
    
    return temp_dist, temp_std

def octahedra_coords_into_bond_vectors(raw,mybox):
    mybox = mybox[:3]
    imybox = 1/mybox
    
    s = np.multiply(imybox,raw)
    bx1 = np.multiply(mybox[:3],s-np.round(s))
            
    bx = bx1/np.mean(np.linalg.norm(bx1,axis=1))
    return bx

def calc_distortions_from_bond_vectors(bx,force_unique=False):
    # constants
    ideal_coords = [[-1, 0,  0],
                    [0, -1,  0],
                    [0,  0, -1],
                    [0,  0,  1],
                    [0,  1,  0],
                    [1,  0,  0]]

    filename_basis = ('.\\basis\\octahedron_basis.json')

    # read json
    try:
        with open(filename_basis, 'r') as f:
            dict_basis = json.load(f)
    except IOError:
        sys.stderr.write('IOError: failed reading from {}.'
                         .format(filename_basis))
        sys.exit(1)
    irrep_distortions = []
    for irrep in dict_basis.keys():
        for elem in dict_basis[irrep]:
            irrep_distortions.append(elem)

    # define "molecules"
    if force_unique: # force one-one correspondance, should only be enabled when orthogonal_frame is on. 
        pymatgen_molecule = pymatgen.core.structure.Molecule(
            species=[Element("Pb"),Element("H"),Element("He"),Element("Li"),Element("Be"),Element("B"),Element("I")],
            coords=np.concatenate((np.zeros((1, 3)), bx), axis=0))
    else:
        pymatgen_molecule = pymatgen.core.structure.Molecule(
            species=[Element("Pb"),Element("I"),Element("I"),Element("I"),Element("I"),Element("I"),Element("I")],
            coords=np.concatenate((np.zeros((1, 3)), bx), axis=0)) 
    
    pymatgen_molecule_ideal = pymatgen.core.structure.Molecule(
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


def match_bx(bx_raw,mybox):
    ideal_coords = np.array([[-1, 0,  0],
                             [0, -1,  0],
                             [0,  0, -1],
                             [0,  0,  1],
                             [0,  1,  0],
                             [1,  0,  0]])
    imybox=1/mybox[:3]
    s = np.multiply(imybox,bx_raw)
    bx = np.multiply(mybox[:3],s-np.round(s))
    bx = bx/np.mean(np.linalg.norm(bx,axis=1))
    order = []
    for ix in range(6):
        fits = np.dot(bx,ideal_coords[ix,:])
        if not (fits[fits.argsort()[-1]] > 0.8 and fits[fits.argsort()[-2]] < 0.6):
            print(bx,fits)
            raise ValueError("The fitting of initial octahedron config to ideal coords is not successful. ")
        order.append(np.argmax(fits))
    assert len(set(order)) == 6 # double-check sanity
    return order

def match_molecules_extra(molecule_transform, molecule_reference):
    # match molecules
    (inds, u, v, rmsd) = pymatgen.analysis.molecule_matcher.HungarianOrderMatcher(
        molecule_reference).match(molecule_transform)

    # affine transform
    molecule_transform.apply_operation(pymatgen.core.operations.SymmOp(
        np.concatenate((
            np.concatenate((u.T, v.reshape(3, 1)), axis=1),
            [np.zeros(4)]), axis=0
        )))
    molecule_transform._sites = np.array(
        molecule_transform._sites)[inds].tolist()

    return molecule_transform,u,rmsd


def calc_displacement(
        pymatgen_molecule, pymatgen_molecule_ideal, irrep_distortions):
    return np.tensordot(
        irrep_distortions,
        (pymatgen_molecule.cart_coords
         - pymatgen_molecule_ideal.cart_coords).ravel()[3:], axes=1)


def match_mixed_halide_octa(bx,hals):
    
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
        conf2r = [Element("Pb"),Element("I"),Element("I"),Element("I"),Element("Br"),Element("Br"),Element("I")]
        conf2p = [Element("Pb"),Element("I"),Element("I"),Element("Br"),Element("Br"),Element("I"),Element("I")]
        
        pymatgen_molecule = pymatgen.core.structure.Molecule(
            species=[Element("Pb"),Element(hals[0]),Element(hals[1]),Element(hals[2]),Element(hals[3]),Element(hals[4]),Element(hals[5])],
            coords=np.concatenate((np.zeros((1, 3)), bx), axis=0))
        pymatgen_molecule_ideal0 = pymatgen.core.structure.Molecule(
            species=conf2r,
            coords=np.concatenate((np.zeros((1, 3)), ideal_coords), axis=0))
        pymatgen_molecule_ideal1 = pymatgen.core.structure.Molecule(
            species=conf2p,
            coords=np.concatenate((np.zeros((1, 3)), ideal_coords), axis=0))
        
        _,_,rmsd0 = match_molecules_extra(
            pymatgen_molecule, pymatgen_molecule_ideal0)
        _,_,rmsd1 = match_molecules_extra(
            pymatgen_molecule, pymatgen_molecule_ideal1)
        
        if rmsd0 > rmsd1: # planar form
            if rmsd0/rmsd1 < distinct_thresh: # tolerance of difference
                raise ValueError(f"The difference between the configurations are not large enough, with a factor of {rmsd0/rmsd1} and threshold is {distinct_thresh}.")
            return (2,1),3 # planar form
        else: # right-angle form
            if rmsd1/rmsd0 < distinct_thresh: # tolerance of difference
                raise ValueError(f"The difference between the configurations are not large enough, with a factor of {rmsd1/rmsd0} and threshold is {distinct_thresh}.")
            return (2,0),2 # right-angle form
    
    elif brnum == 3:
        conf3r = [Element("Pb"),Element("I"),Element("I"),Element("I"),Element("Br"),Element("Br"),Element("Br")]
        conf3p1 = [Element("Pb"),Element("Br"),Element("I"),Element("I"),Element("Br"),Element("I"),Element("Br")]
        conf3p2 = [Element("Pb"),Element("I"),Element("Br"),Element("I"),Element("Br"),Element("Br"),Element("I")]
        conf3p3 = [Element("Pb"),Element("I"),Element("I"),Element("Br"),Element("Br"),Element("Br"),Element("I")]
        
        pymatgen_molecule = pymatgen.core.structure.Molecule(
            species=[Element("Pb"),Element(hals[0]),Element(hals[1]),Element(hals[2]),Element(hals[3]),Element(hals[4]),Element(hals[5])],
            coords=np.concatenate((np.zeros((1, 3)), bx), axis=0))
        pymatgen_molecule_ideal0 = pymatgen.core.structure.Molecule(
            species=conf3r,
            coords=np.concatenate((np.zeros((1, 3)), ideal_coords), axis=0))
        pymatgen_molecule_ideal1 = pymatgen.core.structure.Molecule(
            species=conf3p1,
            coords=np.concatenate((np.zeros((1, 3)), ideal_coords), axis=0))
        pymatgen_molecule_ideal2 = pymatgen.core.structure.Molecule(
            species=conf3p2,
            coords=np.concatenate((np.zeros((1, 3)), ideal_coords), axis=0))
        pymatgen_molecule_ideal3 = pymatgen.core.structure.Molecule(
            species=conf3p3,
            coords=np.concatenate((np.zeros((1, 3)), ideal_coords), axis=0))
        
        _,_,rmsd0 = match_molecules_extra(
            pymatgen_molecule, pymatgen_molecule_ideal0)
        _,_,rmsd1 = match_molecules_extra(
            pymatgen_molecule, pymatgen_molecule_ideal1)
        _,_,rmsd2 = match_molecules_extra(
            pymatgen_molecule, pymatgen_molecule_ideal2)
        _,_,rmsd3 = match_molecules_extra(
            pymatgen_molecule, pymatgen_molecule_ideal3)
        
        rmsd1 = min(rmsd1,rmsd2,rmsd3)
        if rmsd0 > rmsd1: # planar form
            if rmsd0/rmsd1 < distinct_thresh: # tolerance of difference
                raise ValueError(f"The difference between the configurations are not large enough, with a factor of {rmsd0/rmsd1} and threshold is {distinct_thresh}.")
            return (3,1),5 # planar form
        else: # right-angle form
            if rmsd1/rmsd0 < distinct_thresh: # tolerance of difference
                raise ValueError(f"The difference between the configurations are not large enough, with a factor of {rmsd1/rmsd0} and threshold is {distinct_thresh}.")
            return (3,0),4 # right-angle form
        
        #RM = np.array([rmsd0,rmsd1,rmsd2,rmsd3])
        #indtemp = np.argmin(RM)
        #factemp = np.amin(np.delete(RM,np.argmin(RM))/np.amin(RM))
        
        #if indtemp in [1,2,3]: # planar form
        #    if factemp < distinct_thresh: # tolerance of difference
        #        raise ValueError(f"The difference between the configurations are not large enough, with a factor of {factemp} and threshold is {distinct_thresh}.")
        #    return (3,1),5 # planar form
        #else: # right-angle form
        #    if factemp < distinct_thresh: # tolerance of difference
        #        raise ValueError(f"The difference between the configurations are not large enough, with a factor of {factemp} and threshold is {distinct_thresh}.")
        #    return (3,0),4 # right-angle form
    
    elif brnum == 4:
        conf4r = [Element("Pb"),Element("Br"),Element("Br"),Element("Br"),Element("I"),Element("I"),Element("Br")]
        conf4p = [Element("Pb"),Element("Br"),Element("Br"),Element("I"),Element("I"),Element("Br"),Element("Br")]
        
        pymatgen_molecule = pymatgen.core.structure.Molecule(
            species=[Element("Pb"),Element(hals[0]),Element(hals[1]),Element(hals[2]),Element(hals[3]),Element(hals[4]),Element(hals[5])],
            coords=np.concatenate((np.zeros((1, 3)), bx), axis=0))
        pymatgen_molecule_ideal0 = pymatgen.core.structure.Molecule(
            species=conf4r,
            coords=np.concatenate((np.zeros((1, 3)), ideal_coords), axis=0))
        pymatgen_molecule_ideal1 = pymatgen.core.structure.Molecule(
            species=conf4p,
            coords=np.concatenate((np.zeros((1, 3)), ideal_coords), axis=0))
        
        _,_,rmsd0 = match_molecules_extra(
            pymatgen_molecule, pymatgen_molecule_ideal0)
        _,_,rmsd1 = match_molecules_extra(
            pymatgen_molecule, pymatgen_molecule_ideal1)
        
        if rmsd0 > rmsd1: # planar form
            if rmsd0/rmsd1 < distinct_thresh: # tolerance of difference
                raise ValueError(f"The difference between the configurations are not large enough, with a factor of {rmsd0/rmsd1} and threshold is {distinct_thresh}.")
            return (4,1),7 # planar form
        else: # right-angle form
            if rmsd1/rmsd0 < distinct_thresh: # tolerance of difference
                raise ValueError(f"The difference between the configurations are not large enough, with a factor of {rmsd1/rmsd0} and threshold is {distinct_thresh}.")
            return (4,0),6 # right-angle form
    
    elif brnum == 5:
        return (5,0),8

    elif brnum == 6:
        return (6,0),9


def organic_A_site_env(struct,nc):
    from pymatgen.analysis.local_env import CrystalNN
    
    cnn = CrystalNN()
    
    Nlist = []
    Hlist = []
    
    nn1 = 0
    for site in cnn.get_nn_info(struct,n=nc):
        nn1 += 1
        if site['site'].species_string == 'H':
            Hlist.append(site['site_index'])
        elif site['site'].species_string == 'N':
            Nlist.append(site['site_index'])
        else:
            raise TypeError(f"Unexpected species {site['site'].species_string} is found by CrystalNN.")
            
    if nn1 == 3:
        mol_type = 'FA'
    elif nn1 == 4:
        mol_type = 'MA'
    else:
        raise TypeError("Cannot detect the molecule type with CrystalNN.")
    
    
    for N in Nlist:
        nn2 = 0
        for site in cnn.get_nn_info(struct,n=N):
            if site['site'].species_string == 'H':
                Hlist.append(site['site_index'])
                nn2 += 1
        if mol_type == 'FA':
            assert nn2 == 2
        elif mol_type == 'MA':
            assert nn2 == 3
                
    return [nc,sorted(Nlist),sorted(Hlist)], mol_type


def centmass_organic(st0pos,mybox,env):
    
    mybox = mybox[:3]
    imybox = 1/mybox
    
    c = env[0]
    n = env[1]
    h = env[2]
    
    mass = np.zeros((1,3))
    ccc = st0pos[c,:]
    mass = mass + ccc*12
    
    Nvec = st0pos[n,:]-ccc
    Hvec = st0pos[h,:]-ccc

    s = np.multiply(np.expand_dims(imybox, axis=0),Nvec)
    cn = np.multiply(np.expand_dims(mybox, axis=0),s-np.round(s))
    mass = mass + np.sum(ccc + cn,axis=0)*14
    
    s = np.multiply(np.expand_dims(imybox, axis=0),Hvec)
    ch = np.multiply(np.expand_dims(mybox, axis=0),s-np.round(s))
    mass = mass + np.sum(ccc + ch,axis=0)*1
    
    mass = mass/(12+14*len(n)+1*len(h))

    return mass


def centmass_organic_vec(pos,mybox,env):
    
    mybox = mybox[:,:3]
    imybox = 1/mybox
    
    c = env[0]
    n = env[1]
    h = env[2]
    
    mass = np.zeros((pos.shape[0],3))
    ccc = pos[:,c,:]
    mass = mass + ccc*12
    
    Nvec = pos[:,n,:]-np.expand_dims(ccc, axis=1)
    Hvec = pos[:,h,:]-np.expand_dims(ccc, axis=1)

    s = np.multiply(np.expand_dims(imybox, axis=1),Nvec)
    cn = np.multiply(np.expand_dims(mybox, axis=1),s-np.round(s))
    mass = mass + np.sum(np.expand_dims(ccc, axis=1) + cn,axis=1)*14
    
    s = np.multiply(np.expand_dims(imybox, axis=1),Hvec)
    ch = np.multiply(np.expand_dims(mybox, axis=1),s-np.round(s))
    mass = mass + np.sum(np.expand_dims(ccc, axis=1) + ch,axis=1)*1
    
    mass = mass/(12+14*len(n)+1*len(h))

    return mass



def find_B_cage_and_disp(pos,mybox,cent,Bs):
    
    mybox = mybox[:,:3]
    imybox = 1/mybox
    
    B8pos = pos[:,Bs,:]
    BX = B8pos - np.expand_dims(cent, axis=1)
    
    s = np.multiply(np.expand_dims(imybox, axis=1),BX)
    BX = np.multiply(np.expand_dims(mybox, axis=1),s-np.round(s))
    
    Bs = BX + np.expand_dims(cent, axis=1)
    Xcent = np.mean(Bs, axis=1)
    
    disp = cent - Xcent
    
    return disp
    #return np.expand_dims(disp, axis=1)




