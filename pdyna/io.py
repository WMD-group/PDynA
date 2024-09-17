"""
pdyna.io: The collection of I/O functions to various input formats.

"""

import re
import numpy as np
from ase.atoms import Atoms
from ase.calculators.lammps import convert


def read_lammps_dir(fdir,allow_multi=False):
    """
    Read the LAMMPS input files in the directory and extract the simulation settings.

    Args:
        fdir (str): The directory path.
        allow_multi (bool): Allow multiple LAMMPS input files in the directory.

    Returns:
        dict or list of dict: The simulation settings.
    """

    from glob import glob
    filelist = glob(fdir+"*.in")
    if len(filelist) == 0:
        raise FileNotFoundError("Can't find any LAMMPS .in file in the directory.")
        
    if not allow_multi:
        if len(filelist) > 1:
            raise FileExistsError("There are more than one LAMMPS .in file in the directory.")
        return read_lammps_settings(filelist[0])
    else:
        return [read_lammps_settings(f) for f in filelist]
    

def read_lammps_settings(infile): # internal use 
    """
    Read the LAMMPS input file and extract the simulation settings.

    Args:
        infile (str): The LAMMPS input file.

    Returns:
        dict: The simulation settings
    """

    with open(infile,"r") as fp:
        lines = fp.readlines()
    tvelo = 0
    ti = None
    tf = None
    tstep = None
    nsw = 0
    for line in lines:
        if line.startswith("velocity"):
            tvelo = float(line.split()[3])
        if line.startswith("fix"):
            ti = float(line.split()[5])
            tf = float(line.split()[6])
        if line.startswith("timestep"):
            tstep = float(line.split()[1])*1000
        if line.startswith("run"):
            nsw = float(line.split()[1])
    
    #if tvelo != ti:
    #    print("!Lammps in file: the thermalization temperature is different from the initial temperature.")

    return {"Ti":ti, "Tf":tf, "tstep":tstep, "nsw":nsw}    


def read_ase_md_settings(fdir): # internal use 
    """
    Read the ASE MD input file and extract the simulation settings.

    Args:
        fdir (str): The directory path.

    Returns:
        tuple: The simulation settings
    """

    from glob import glob

    filelist = glob(fdir+"*.py")
    if len(filelist) == 0:
        raise FileNotFoundError("Can't find any ASE MD .py file in the directory.")
    if len(filelist) > 1:
        raise FileExistsError("There are more than one ASE MD .py file in the directory.")
    
    infile = filelist[0]

    with open(infile,"r") as fp:
        lines = fp.readlines()
    
    ti = None
    tstep = None
    nsw = None
    nblock = None
    for line in lines:
        if line.startswith("dyn = NPT"):
            sets = line.split(',')
            for s in sets:
                if "temperature_K" in s:
                    ti = int(s.split("=")[1])
            tstep=float(sets[1].rstrip("*units.fs"))
        if line.startswith("dyn.run"):
            nsw = int(re.search(r'\d+', line).group())
        if line.startswith("dyn.attach") and "write_frame" in line:
            nblock = int(re.search(r'\d+', line).group())
    
    return ti, tstep, nsw, nblock

    
def process_lat(m):
    """ 
    Convert lattice matrix to abc and three angles.  

    Args:
        m (numpy.ndarray): The lattice matrix.

    Returns:
        numpy.ndarray: The abc lattice vectors and three angles.
    """
    
    abc = np.sqrt(np.sum(m**2, axis=1))
    angles = np.zeros(3)
    for i in range(3):
        j = (i + 1) % 3
        k = (i + 2) % 3
        angles[i] = np.clip(np.dot(m[j], m[k])/(abc[j] * abc[k]),-1,1)
    angles = np.arccos(angles) * 180.0 / np.pi
    return np.concatenate((abc,angles)).reshape(1,6)


def process_lat_reverse(cellpar):
    """ 
    Convert abc and three angles to lattice matrix.  
    Modified from ASE functions.

    Args:
        cellpar (numpy.ndarray): The abc lattice vectors and three angles.  

    Returns:
        numpy.ndarray: The lattice matrix.
    """

    X = np.array([1., 0., 0.])
    Y = np.array([0., 1., 0.])
    Z = np.array([0., 0., 1.])

    # Express va, vb and vc in the X,Y,Z-system
    alpha, beta, gamma = 90., 90., 90.
    if isinstance(cellpar, (int, float)):
        a = b = c = cellpar
    elif len(cellpar) == 1:
        a = b = c = cellpar[0]
    elif len(cellpar) == 3:
        a, b, c = cellpar
    else:
        a, b, c, alpha, beta, gamma = cellpar

    # Handle orthorhombic cells separately to avoid rounding errors
    eps = 2 * np.spacing(90.0, dtype=np.float64)  # around 1.4e-14
    # alpha
    if abs(abs(alpha) - 90) < eps:
        cos_alpha = 0.0
    else:
        cos_alpha = np.cos(alpha * np.pi / 180.0)
    # beta
    if abs(abs(beta) - 90) < eps:
        cos_beta = 0.0
    else:
        cos_beta = np.cos(beta * np.pi / 180.0)
    # gamma
    if abs(gamma - 90) < eps:
        cos_gamma = 0.0
        sin_gamma = 1.0
    elif abs(gamma + 90) < eps:
        cos_gamma = 0.0
        sin_gamma = -1.0
    else:
        cos_gamma = np.cos(gamma * np.pi / 180.0)
        sin_gamma = np.sin(gamma * np.pi / 180.0)

    # Build the cell vectors
    va = a * np.array([1, 0, 0])
    vb = b * np.array([cos_gamma, sin_gamma, 0])
    cx = cos_beta
    cy = (cos_alpha - cos_beta * cos_gamma) / sin_gamma
    cz_sqr = 1. - cx * cx - cy * cy
    assert cz_sqr >= 0
    cz = np.sqrt(cz_sqr)
    vc = c * np.array([cx, cy, cz])

    # Convert to the Cartesian x,y,z-system
    abc = np.vstack((va, vb, vc))
    T = np.vstack((X, Y, Z))
    cell = np.dot(abc, T)

    return cell


def read_xdatcar(filename,natom):
    """
    Read the VASP XDATCAR file and extract the atomic symbols, lattices, and atomic positions.

    Args:
        filename (str): The XDATCAR file.
        natom (int): The number of atoms.

    Returns:
        tuple: The atomic symbols, lattices, and atomic positions.
    """

    import warnings
    from monty.io import zopen
    from pymatgen.util.io_utils import clean_lines
    from pymatgen.core.periodic_table import Element
    
    from pdyna.structural import get_cart_from_frac, get_frac_from_cart

    def from_string(data):
        """
        Modified from the Pymatgen function Poscar.from_string

        Args:
            data (str): String representation of VASP structure.

        Returns:
            tuple: The atomic symbols, lattices, and atomic positions.
        """
        # "^\s*$" doesn't match lines with no whitespace
        chunks = re.split(r"\n\s*\n", data.rstrip(), flags=re.MULTILINE)
        try:
            if chunks[0] == "":
                chunks.pop(0)
                chunks[0] = "\n" + chunks[0]
        except IndexError:
            raise ValueError("Empty structure")

        # Parse positions
        lines = tuple(clean_lines(chunks[0].split("\n"), False))
        scale = float(lines[1])
        lattice = np.array([[float(i) for i in line.split()] for line in lines[2:5]])
        if scale < 0:
            # In vasp, a negative scale factor is treated as a volume. We need
            # to translate this to a proper lattice vector scaling.
            vol = abs(np.linalg.det(lattice))
            lattice *= (-scale / vol) ** (1 / 3)
        else:
            lattice *= scale

        vasp5_symbols = False
        try:
            natoms = [int(i) for i in lines[5].split()]
            ipos = 6
        except ValueError:
            vasp5_symbols = True
            symbols = lines[5].split()

            nlines_symbols = 1
            for nlines_symbols in range(1, 11):
                try:
                    int(lines[5 + nlines_symbols].split()[0])
                    break
                except ValueError:
                    pass
            for iline_symbols in range(6, 5 + nlines_symbols):
                symbols.extend(lines[iline_symbols].split())
            natoms = []
            iline_natoms_start = 5 + nlines_symbols
            for iline_natoms in range(iline_natoms_start, iline_natoms_start + nlines_symbols):
                natoms.extend([int(i) for i in lines[iline_natoms].split()])
            atomic_symbols = []
            for i, nat in enumerate(natoms):
                atomic_symbols.extend([symbols[i]] * nat)
            ipos = 5 + 2 * nlines_symbols

        pos_type = lines[ipos].split()[0]

        has_selective_dynamics = False
        # Selective dynamics
        if pos_type[0] in "sS":
            has_selective_dynamics = True
            ipos += 1
            pos_type = lines[ipos].split()[0]

        cart = pos_type[0] in "cCkK"
        n_sites = sum(natoms)

        if not vasp5_symbols:
            ind = 3 if not has_selective_dynamics else 6
            try:
                # Check if names are appended at the end of the coordinates.
                atomic_symbols = [l.split()[ind] for l in lines[ipos + 1 : ipos + 1 + n_sites]]
                # Ensure symbols are valid elements
                if not all(Element.is_valid_symbol(sym) for sym in atomic_symbols):
                    raise ValueError("Non-valid symbols detected.")
                vasp5_symbols = True
            except (ValueError, IndexError):
                # Defaulting to false names.
                atomic_symbols = []
                for i, nat in enumerate(natoms):
                    sym = Element.from_Z(i + 1).symbol
                    atomic_symbols.extend([sym] * nat)
                warnings.warn(f"Elements in POSCAR cannot be determined. Defaulting to false names {atomic_symbols}.")

        # read the atomic coordinates
        coords = []
        selective_dynamics = [] if has_selective_dynamics else None
        for i in range(n_sites):
            toks = lines[ipos + 1 + i].split()
            crd_scale = scale if cart else 1
            coords.append([float(j) * crd_scale for j in toks[:3]])
            if has_selective_dynamics:
                selective_dynamics.append([tok.upper()[0] == "T" for tok in toks[3:6]])


        l6 = process_lat(lattice)
        atomic_symbols,
        lattice,
        coords = np.array(coords)
        if cart:
            cart_coords = coords
            frac_coords = get_frac_from_cart(coords, lattice)
        else:
            frac_coords = coords
            cart_coords = get_cart_from_frac(coords, lattice)
            
        return atomic_symbols, lattice, l6, frac_coords, cart_coords       

    preamble = None
    coords_str = []
    preamble_done = False

    f = zopen(filename, "rt")
    fcount = 0
    Allpos = np.empty((0,natom,3))
    lattice = np.empty((0,6))
    latmat = np.empty((0,3,3))

    for l in f:
        new_frame = False
        l = l.strip()
        if preamble is None:
            preamble = [l]
            title = l
        elif title == l:
            preamble_done = False
            p = "\n".join(preamble + ["Direct"] + coords_str)
            new_frame = True #texts.append(p)
            coords_str = []
            preamble = [l]
        elif not preamble_done:
            if l == "" or "Direct configuration=" in l:
                preamble_done = True
                tmp_preamble = [preamble[0]]
                for i in range(1, len(preamble)):
                    if preamble[0] != preamble[i]:
                        tmp_preamble.append(preamble[i])
                    else:
                        break
                preamble = tmp_preamble
            else:
                preamble.append(l)
        elif l == "" or "Direct configuration=" in l:
            p = "\n".join(preamble + ["Direct"] + coords_str)
            new_frame = True #texts.append(p)
            coords_str = []
        else:
            coords_str.append(l)
        if new_frame:
            fcount += 1
            atomic_symbols, lat, l6, frac_coords, cart_coords = from_string(p)
            Allpos = np.concatenate((Allpos,cart_coords[np.newaxis,:]),axis=0)
            lattice = np.concatenate((lattice,l6),axis=0)
            latmat = np.concatenate((latmat,lat[np.newaxis,:]),axis=0)
            
    p = "\n".join(preamble + ["Direct"] + coords_str)
    fcount += 1
    atomic_symbols, lat, l6, frac_coords, cart_coords = from_string(p)
    Allpos = np.concatenate((Allpos,cart_coords[np.newaxis,:]),axis=0)
    lattice = np.concatenate((lattice,l6),axis=0)
    latmat = np.concatenate((latmat,lat[np.newaxis,:]),axis=0)

    assert Allpos.shape[0] == fcount
    assert lattice.shape[0] == fcount
    assert latmat.shape[0] == fcount
    
    return atomic_symbols, lattice, latmat, Allpos


def read_lammps_dump(filepath,specorder=None): 
    """
    Read the LAMMPS trajectory file and extract the atomic symbols, lattices, and atomic positions.    
    Modified from ASE LAMMPS reading functions

    Args:
        filepath (str): The LAMMPS trajectory file.
        specorder (list): The order of atomic species.

    Returns:
        tuple: The atomic symbols, lattices, and atomic positions.
    """
    from collections import deque
    from pymatgen.io.ase import AseAtomsAdaptor as aaa
    # Load all dumped timesteps into memory simultaneously
    fp = open(filepath,"r")
    lines = deque(fp.readlines())
    index_end = -1

    n_atoms = 0
    framenums = []
    Allpos_list = []
    lattice = np.empty((0,6))
    latmat = np.empty((0,3,3))
    asymb = 0

    # avoid references before assignment in case of incorrect file structure
    cell, celldisp = None, None

    while len(lines) > n_atoms:
        line = lines.popleft()

        if "ITEM: TIMESTEP" in line:
            n_atoms = 0
            line = lines.popleft()
            stepnum = int(line.split()[0])

        if "ITEM: NUMBER OF ATOMS" in line:
            line = lines.popleft()
            n_atoms = int(line.split()[0])

        if "ITEM: BOX BOUNDS" in line:
            tilt_items = line.split()[3:]
            celldatarows = [lines.popleft() for _ in range(3)]
            celldata = np.loadtxt(celldatarows)
            diagdisp = celldata[:,:2].reshape(6, 1).flatten()

            # determine cell tilt (triclinic case)
            if len(celldata[0]) > 2:
                offdiag = celldata[:, 2]
                if len(tilt_items) >= 3:
                    sort_index = [tilt_items.index(i)
                                  for i in ["xy", "xz", "yz"]]
                    offdiag = offdiag[sort_index]
            else:
                offdiag = (0.0,) * 3

            cell, celldisp = construct_cell(diagdisp, offdiag)

        if "ITEM: ATOMS" in line:
            colnames = line.split()[2:]
            datarows = [lines.popleft() for _ in range(n_atoms)]
            data = np.loadtxt(datarows, dtype=str)
            atomic_symbols, lm, l6, frac_coords, cart_coords = process_lammps_data(data,colnames,cell,celldisp,specorder=specorder)
            if asymb == 0:
                asymb = atomic_symbols
            framenums.append(stepnum)
            
            Allpos_list.append(cart_coords)
            lattice = np.concatenate((lattice,l6),axis=0)
            latmat = np.concatenate((latmat,lm[np.newaxis,:]),axis=0)

        if lattice.shape[0] > index_end >= 0:
            break
    
    if Allpos_list[0].shape != Allpos_list[-1].shape: # the last block is incomplete
        for istop in range(len(Allpos_list)):
            nafr = Allpos_list[istop].shape
            if nafr != Allpos_list[0].shape: break
        Allpos = np.array(Allpos_list[:istop])
        # isolate the first frame which is the initial structure as st0
        pos0 = Allpos[0,:]
        cell = latmat[0,:]
        Allpos = Allpos[1:istop,:]
        lattice = lattice[1:istop,:]
        latmat = latmat[1:istop,:]
    else:
        Allpos = np.array(Allpos_list)
        # isolate the first frame which is the initial structure as st0
        pos0 = Allpos[0,:]
        cell = latmat[0,:]
        Allpos = Allpos[1:,:]
        lattice = lattice[1:,:]
        latmat = latmat[1:,:]
    
    assert Allpos.shape[0] == lattice.shape[0] == latmat.shape[0]
    
    out_atoms = Atoms(symbols=np.array(asymb),positions=pos0,pbc=[True,True,True],celldisp=celldisp,cell=cell)
    st0 = aaa.get_structure(out_atoms)
    
    maxframe = framenums[-1]
    if framenums[1]-framenums[0] > framenums[-1]-framenums[-2]:
        maxframe -= (framenums[1]-framenums[0])
    
    return asymb, lattice, latmat, Allpos, st0, maxframe, framenums[-1]-framenums[-2]


def read_xyz(filepath): 
    """
    Read the XYZ format trajectory file and extract the atomic symbols, lattices, and atomic positions.    
    Modified from Pymatgen xyz reading functions

    Args:
        filepath (str): The XYZ format file.

    Returns:
        tuple: The atomic symbols, lattices, and atomic positions.
    """

    from pymatgen.io.ase import AseAtomsAdaptor as aaa
    
    def read_xyz_block(bloc):
        """ 
        Read the XYZ frame block and extract the atomic symbols, lattices, and atomic positions.

        Args:
            bloc (list): The XYZ frame block.
            
        Returns:
            tuple: The atomic symbols, lattices, and atomic positions.
        """

        num_sites = int(bloc[0])
        if len(bloc) != num_sites+2:
            raise SyntaxError("The XYZ format should be line 1: N-atoms; line 2: cell dimension: [a b c alpha beta gamma]; line 3-end: species and atomic coordinates. ")
        cellstr = bloc[1]
        try: 
            cell = [float(entry) for entry in cellstr.split()]
        except ValueError:
            raise SyntaxError("The XYZ format should be line 1: N-atoms; line 2: cell dimension: [a b c alpha beta gamma]; line 3-end: species and atomic coordinates. ")
        
        coords = []
        sp = []
        coord_patt = re.compile(r"(\w+)\s+([0-9\-\+\.*^eEdD]+)\s+([0-9\-\+\.*^eEdD]+)\s+([0-9\-\+\.*^eEdD]+)")
        for i in range(2, 2 + num_sites):
            m = coord_patt.search(bloc[i])
            if m:
                sp.append(m.group(1))  # this is 1-indexed
                xyz = [val.lower().replace("d", "e").replace("*^", "e") for val in m.groups()[1:4]]
                coords.append([float(val) for val in xyz])
        return sp, np.array(coords), np.array(cell)
    
    contents = open(filepath, "rt").read()
    lines = re.split("\n", contents)
    blockheads = [i for i,s in enumerate(lines) if bool(re.match("\s*\d+\s*$", s))]
    if len(blockheads) < 2: 
        raise ValueError("The frames can be read correctly, please check the file integrity.")
    frames = []
    cart = []
    celldim = []
    cellmat = []
    for i,j in enumerate(blockheads):
        if j != blockheads[-1]:
            bloc = lines[j:blockheads[i+1]]
        else:
            bloc = lines[j:]
            while len(bloc[-1]) == 0:
                bloc.pop()
                
        asymb, ci, celli = read_xyz_block(bloc)
        cart.append(ci)
        celldim.append(celli)
        cellmat.append(process_lat_reverse(celli))
    
    Allpos = np.array(cart)
    lattice = np.array(celldim)
    latmat = np.array(cellmat)    
        
    assert Allpos.shape[0] == lattice.shape[0] == latmat.shape[0]
    
    # isolate the first frame which is the initial structure as st0
    Allpos = Allpos[1:,:]
    lattice = lattice[1:,:]
    latmat = latmat[1:,:]
        
    out_atoms = Atoms(symbols=np.array(asymb),positions=Allpos[0,:],pbc=[True,True,True],celldisp=np.array([0., 0., 0.]),cell=latmat[0,:])
    st0 = aaa.get_structure(out_atoms)
    
    return asymb, lattice, latmat, Allpos, st0, latmat.shape[0]


def read_pdb(filepath): 
    """
    Read the protein data bank (PDB) trajectory file and extract the atomic symbols, lattices, and atomic positions.    
    Modified from Pymatgen xyz reading functions

    Args:
        filepath (str): The PDB file.

    Returns:
        tuple: The atomic symbols, lattices, and atomic positions.
    """
    from pymatgen.io.ase import AseAtomsAdaptor as aaa
    
    def read_pdb_block(bloc):
        """
        Read the PDB frame block and extract the atomic symbols, lattices, and atomic positions.

        Args:
            bloc (list): The PDB frame block.

        Returns:
            tuple: The atomic symbols, lattices, and atomic positions.
        """
        
        coords = []
        sp = []
        for line in bloc:
            if line.startswith("CRYST1"):
                str1 = line.lstrip("CRYST1").split()
                cell = np.array([float(e) for e in str1])
            if line.startswith("ATOM"):
                coords.append(np.array([float(line[30:38]),float(line[38:46]),float(line[46:54])], dtype=np.float64))
                sp.append(line[76:78].strip())
        
        return sp, np.array(coords), cell
    
    contents = open(filepath, "rt").read()
    lines = re.split("\n", contents)
    b0 = []
    b1 = []
    for i,s in enumerate(lines):
        if s.startswith("REMARK"): b0.append(i)
        elif s.startswith("END"): b1.append(i)
    
    if len(b0) != len(b1):
        raise ValueError("The PDB file format is not recognized, please check file integrity. ")
    if len(b0) < 2:
        raise ValueError("The PDB file format is not recognized, please check file integrity. ")

    cart = []
    celldim = []
    cellmat = []
    for i in range(len(b0)):
        bloc = lines[b0[i]:b1[i]]
        
        asymb, ci, celli = read_pdb_block(bloc)
        cart.append(ci)
        celldim.append(celli)
        cellmat.append(process_lat_reverse(celli))
    
    Allpos = np.array(cart)
    lattice = np.array(celldim)
    latmat = np.array(cellmat)    
        
    assert Allpos.shape[0] == lattice.shape[0] == latmat.shape[0]
    
    # isolate the first frame which is the initial structure as st0
    pos0 = Allpos[0,:]
    lat0 = latmat[0,:]
    Allpos = Allpos[1:,:]
    lattice = lattice[1:,:]
    latmat = latmat[1:,:]
        
    out_atoms = Atoms(symbols=np.array(asymb),positions=pos0,pbc=[True,True,True],celldisp=np.array([0., 0., 0.]),cell=lat0)
    st0 = aaa.get_structure(out_atoms)
    
    return asymb, lattice, latmat, Allpos, st0, latmat.shape[0]


def read_ase_traj(filepath): 
    """
    Read the ASE internal trajectory file and extract the atomic symbols, lattices, and atomic positions.    
    Modified from ASE original trajectory reading functions

    Args:
        filepath (str): The ASE trajectory file.

    Returns:
        tuple: The atomic symbols, lattices, and atomic positions
    """
    
    from ase.io import Trajectory
    from pymatgen.io.ase import AseAtomsAdaptor as aaa
    
    contents = Trajectory(filepath)
    
    cart = []
    celldim = []
    cellmat = []
    for bloc in contents:
        ci = bloc.positions
        cmat = bloc.cell.array
        
        cart.append(ci)
        celldim.append(process_lat(cmat)[0,:])
        cellmat.append(cmat)
    
    Allpos = np.array(cart)
    lattice = np.array(celldim)
    latmat = np.array(cellmat)    
        
    assert Allpos.shape[0] == lattice.shape[0] == latmat.shape[0]
    
    # isolate the first frame which is the initial structure as st0
    Allpos = Allpos[1:,:]
    lattice = lattice[1:,:]
    latmat = latmat[1:,:]
        
    st0 = aaa.get_structure(contents[0])
    if not np.array_equal(np.array(st0.atomic_numbers),np.array(contents[0].numbers)):
        raise TypeError("Fatal: the converted Pymatgen structure does not match with the ASE Atoms. ")
    asymb = []
    for s in st0.species:
        asymb.append(s.name)
    
    return asymb, lattice, latmat, Allpos, st0, latmat.shape[0]


def process_lammps_data(
    data,
    colnames,
    cell,
    celldisp,
    order=True,
    specorder=None,
    units="metal",):
    """
    Extract positions and other per-atom parameters from block of LAMMPS data.
    Modified from ASE functions

    Args:
        data (numpy.ndarray): The LAMMPS data block.
        colnames (list): The column names.
        cell (numpy.ndarray): The lattice matrix.
        celldisp (numpy.ndarray): The lattice displacement.
        order (bool): Order the atoms.
        specorder (list): The order of atomic species.
        units (str): The LAMMPS units.

    Returns:
        tuple: The atomic symbols, lattices, and atomic positions.
    """
    
    from pdyna.structural import get_frac_from_cart, get_cart_from_frac

    # read IDs if given and order if needed
    if "id" in colnames:
        ids = data[:, colnames.index("id")].astype(int)
        if order:
            sort_order = np.argsort(ids)
            data = data[sort_order, :]

    # determine the elements
    if "element" in colnames:
        elements = data[:, colnames.index("element")]
    elif "type" in colnames:
        elements = data[:, colnames.index("type")].astype(int)
        if specorder:
            elements = [specorder[t - 1] for t in elements]
    else:
        raise ValueError("Cannot determine atom types form LAMMPS dump file")

    def get_quantity(labels, quantity=None):
        """ 
        Extract the quantity from the LAMMPS data block.
        """
        try:
            cols = [colnames.index(label) for label in labels]
            if quantity:
                return convert(data[:, cols].astype(float), quantity,
                               units, "ASE")

            return data[:, cols].astype(float)
        except ValueError:
            return None

    # Positions
    positions = None
    scaled_positions = None
    if "x" in colnames:
        # doc: x, y, z = unscaled atom coordinates
        positions = get_quantity(["x", "y", "z"], "distance")
    elif "xs" in colnames:
        # doc: xs,ys,zs = scaled atom coordinates
        scaled_positions = get_quantity(["xs", "ys", "zs"])
    elif "xu" in colnames:
        # doc: xu,yu,zu = unwrapped atom coordinates
        positions = get_quantity(["xu", "yu", "zu"], "distance")
    elif "xsu" in colnames:
        # xsu,ysu,zsu = scaled unwrapped atom coordinates
        scaled_positions = get_quantity(["xsu", "ysu", "zsu"])
    else:
        raise ValueError("No atomic positions found in LAMMPS output")

    # convert cell
    cell = convert(cell, "distance", units, "ASE")
    celldisp = convert(celldisp, "distance", units, "ASE")
    
    l6 = process_lat(cell)
    
    if positions is not None:
        cart_coords = positions - celldisp
        frac_coords = get_frac_from_cart(cart_coords,cell)
    else:
        frac_coords = scaled_positions
        cart_coords = get_cart_from_frac(frac_coords, cell) - celldisp

    return list(elements), cell, l6, frac_coords, cart_coords  


def lammps_data_to_ase_atoms( # deprecated
    data,
    colnames,
    cell,
    celldisp,
    pbc=False,
    atomsobj=Atoms,
    order=True,
    specorder=None,
    prismobj=None,
    units="metal",
):
    """
    Extract positions and other per-atom parameters and create Atoms
    Taken directly from ASE

    Args:
        data (numpy.ndarray): The LAMMPS data block.
        colnames (list): The column names.
        cell (numpy.ndarray): The lattice matrix.
        celldisp (numpy.ndarray): The lattice displacement.
        pbc (bool): The periodic boundary conditions.
        atomsobj (class): The atoms object class.
        order (bool): Order the atoms.
        specorder (list): The order of atomic species.
        prismobj (class): The prism object.

    Returns:
        ase.Atoms: The atomic structure in ASE.Atoms format.
    """
    
    from ase.calculators.singlepoint import SinglePointCalculator

    # read IDs if given and order if needed
    if "id" in colnames:
        ids = data[:, colnames.index("id")].astype(int)
        if order:
            sort_order = np.argsort(ids)
            data = data[sort_order, :]

    # determine the elements
    if "element" in colnames:
        # priority to elements written in file
        elements = data[:, colnames.index("element")]
    elif "type" in colnames:
        # fall back to `types` otherwise
        elements = data[:, colnames.index("type")].astype(int)

        # reconstruct types from given specorder
        if specorder:
            elements = [specorder[t - 1] for t in elements]
    else:
        # todo: what if specorder give but no types?
        # in principle the masses could work for atoms, but that needs
        # lots of cases and new code I guess
        raise ValueError("Cannot determine atom types form LAMMPS dump file")

    def get_quantity(labels, quantity=None):
        """
        Extract the quantity from the LAMMPS data block.
        """
        try:
            cols = [colnames.index(label) for label in labels]
            if quantity:
                return convert(data[:, cols].astype(float), quantity,
                               units, "ASE")

            return data[:, cols].astype(float)
        except ValueError:
            return None

    # Positions
    positions = None
    scaled_positions = None
    if "x" in colnames:
        # doc: x, y, z = unscaled atom coordinates
        positions = get_quantity(["x", "y", "z"], "distance")
    elif "xs" in colnames:
        # doc: xs,ys,zs = scaled atom coordinates
        scaled_positions = get_quantity(["xs", "ys", "zs"])
    elif "xu" in colnames:
        # doc: xu,yu,zu = unwrapped atom coordinates
        positions = get_quantity(["xu", "yu", "zu"], "distance")
    elif "xsu" in colnames:
        # xsu,ysu,zsu = scaled unwrapped atom coordinates
        scaled_positions = get_quantity(["xsu", "ysu", "zsu"])
    else:
        raise ValueError("No atomic positions found in LAMMPS output")

    velocities = get_quantity(["vx", "vy", "vz"], "velocity")
    charges = get_quantity(["q"], "charge")
    forces = get_quantity(["fx", "fy", "fz"], "force")
    quaternions = get_quantity(["c_q[1]", "c_q[2]", "c_q[3]", "c_q[4]"]) # not tested

    # convert cell
    cell = convert(cell, "distance", units, "ASE")
    celldisp = convert(celldisp, "distance", units, "ASE")
    if prismobj:
        celldisp = prismobj.vector_to_ase(celldisp)
        cell = prismobj.update_cell(cell)

    if quaternions:
        from ase.quaternions import Quaternions
        out_atoms = Quaternions(
            symbols=elements,
            positions=positions,
            cell=cell,
            celldisp=celldisp,
            pbc=pbc,
            quaternions=quaternions,
        )
    elif positions is not None:
        # reverse coordinations transform to lammps system
        # (for all vectors = pos, vel, force)
        if prismobj:
            positions = prismobj.vector_to_ase(positions, wrap=True)

        out_atoms = atomsobj(
            symbols=elements,
            positions=positions,
            pbc=pbc,
            celldisp=celldisp,
            cell=cell
        )
    elif scaled_positions is not None:
        out_atoms = atomsobj(
            symbols=elements,
            scaled_positions=scaled_positions,
            pbc=pbc,
            celldisp=celldisp,
            cell=cell,
        )

    if velocities is not None:
        if prismobj:
            velocities = prismobj.vector_to_ase(velocities)
        out_atoms.set_velocities(velocities)
    if charges is not None:
        out_atoms.set_initial_charges(charges)
    if forces is not None:
        if prismobj:
            forces = prismobj.vector_to_ase(forces)
        # !TODO: use another calculator if available (or move forces
        #        to atoms.property) (other problem: synchronizing
        #        parallel runs)
        calculator = SinglePointCalculator(out_atoms, energy=0.0,
                                           forces=forces)
        out_atoms.calc = calculator

    # process the extra columns of fixes, variables and computes
    #    that can be dumped, add as additional arrays to atoms object
    for colname in colnames:
        # determine if it is a compute or fix (but not the quaternian)
        if (colname.startswith('f_') or colname.startswith('v_') or
                (colname.startswith('c_') and not colname.startswith('c_q['))):
            out_atoms.new_array(colname, get_quantity([colname]),
                                dtype='float')

    return out_atoms

def get_max_index(index):
    """Get the maximum index from a slice object or integer."""
    if np.isscalar(index):
        return index
    elif isinstance(index, slice):
        return index.stop if (index.stop is not None) else float("inf")

def construct_cell(diagdisp, offdiag):
    """
    Help function to create an ASE-cell with displacement vector from the lammps coordination system parameters.

    Args:
        diagdisp (tuple): The diagonal displacement vector.
        offdiag (tuple): The off-diagonal displacement vector.

    Returns:
        tuple: The cell matrix and the displacement vector.
    """
    xlo, xhi, ylo, yhi, zlo, zhi = diagdisp
    xy, xz, yz = offdiag

    # create ase-cell from lammps-box
    xhilo = (xhi - xlo) - abs(xy) - abs(xz)
    yhilo = (yhi - ylo) - abs(yz)
    zhilo = zhi - zlo
    celldispx = xlo - min(0, xy) - min(0, xz)
    celldispy = ylo - min(0, yz)
    celldispz = zlo
    cell = np.array([[xhilo, 0, 0], [xy, yhilo, 0], [xz, yz, zhilo]])
    celldisp = np.array([celldispx, celldispy, celldispz])

    return cell, celldisp


def chemical_from_formula(struct):
    """
    Get the chemical formula from the structure object.
    Only recorded a limited range of structures.

    Args:
        struct (Structure): The structure object.

    Returns:
        str: The chemical formula.
    """
    chem_lib = {'CsPbI3': r'CsPbI$_{3}$', 
                'CsPbBr3': r'CsPbBr$_{3}$',
                'H5PbCI3N2': r'FAPbI$_{3}$',
                'H5PbCBr3N2': r'FAPbBr$_{3}$',
                'H6PbCI3N': r'MAPbI$_{3}$',
                'H6PbCBr3N': r'MAPbBr$_{3}$'}
    
    if struct.composition.reduced_formula in chem_lib:
        return chem_lib[struct.composition.reduced_formula]
    else:
        return struct.composition.reduced_formula
    

def print_time(times):
    """
    Print the elapsed time for each step in the calculation.
    """
    import time
    def time_format(secs):
        """ Format the time in hours, minutes, and seconds. """
        return time.strftime("%H:%M:%S", time.gmtime(secs))
    
    time_quantities = {'env_resolve':         "Structure Resolving:   {}",
                       'lattice':             "Lattice Parameter:     {}",
                       'tavg':                "Time-averaging:        {}",
                       'tilt_distort':        "Tilting & Distortion:  {}",
                       'MO':                  "Molecular Orientation: {}",
                       'RDF':                 "Radial Distribution:   {}",
                       'site_disp':           "A-site Displacement:   {}",
                       'property_processing': "Property processing:   {}",
                       }
    
    print("--Elapsed Time")
    print("Data Reading:          {}".format(time_format(round(times['reading']))))
    for printkey, printstr in time_quantities.items():
        if printkey in times:
            print(printstr.format(time_format(round(times[printkey]))))
    print("Total:                 {}".format(time_format(round(times['total']))))
    
    
def display_A_sites(A_sites):
    """
    Display the A-site occupancy in the structure.
    """
    prstr = []
    for key in A_sites:
        sites = len(A_sites[key])
        if sites > 0:
            prstr.append(key+": "+str(sites))
    print("A-sites are ->",", ".join(prstr))
