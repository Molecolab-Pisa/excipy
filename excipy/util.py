import os
from collections import defaultdict
from functools import partial
from itertools import product as iterproduct
from itertools import zip_longest
import numpy as np
from numpy import pi as PI
from tqdm import tqdm
import h5py


# =============================================================================
# Constants
# =============================================================================


ELECTRON_CHARGE = 1.602176634e-19
SPEED_OF_LIGHT = 299792458.0
MAGNETIC_CONSTANT = 1.25663706212e-06
ELECTRIC_CONSTANT = 8.8541878128e-12
PLANCK = 6.62607015e-34

ANG2BOHR = 1.889725989

COULOMB_CONSTANT = 1.0 / (4.0 * PI * ELECTRIC_CONSTANT)
JOULE2CM_1 = 1.0 / (PLANCK * SPEED_OF_LIGHT * 1e2)
HARTREE2COULOMB = ELECTRON_CHARGE
HARTREESQ_NM2CM_1 = HARTREE2COULOMB**2 * COULOMB_CONSTANT * JOULE2CM_1 * 1e9
HARTREE2CM_1 = 219474.63
EV2CM = 8065.544


# =============================================================================
# Misc
# =============================================================================


# Progress bar
pbar = partial(tqdm, colour="green")


# =============================================================================
# Helpers
# =============================================================================


atomsym2atomnum = {
    "H": 1,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Mg": 12,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Zn": 30,
    "Br": 35,
    "I": 53,
}


atomnum2atomsym = {value: key for key, value in atomsym2atomnum.items()}


# =============================================================================
# Functions to compute geometric factors
# =============================================================================


def _distance_geometric_center(coords1, coords2):
    """
    Compute the distance between the geometric centers
    of the given coordinates.
    Arguments
    ---------
    coords1   : ndarray, (num_frames, num_atoms, 3)
    coords2   : ndarray, (num_frames, num_atoms, 3)
    Returns
    -------
    distance  : ndarray, (num_frames,)
    """
    center1 = np.mean(coords1, axis=1)
    center2 = np.mean(coords2, axis=1)
    distance = np.sum((center1 - center2) ** 2, axis=-1) ** 0.5
    return distance


def _compute_retain_list(coords, residue_ids, pairs, cutoff):
    """
    Compute a list of pairs that have to be retained (True)
    of excluded (False) on the basis of a distance cutoff
    between the geometric centers of the two molecules
    Arguments
    ---------
    coords      : list of ndarray, (num_frames, num_atoms, 3)
                List of atomic coordinates, each entry corresponds to a molecule
    residue_ids : list of str
                Indeces/names of the residue_ids (used for printing only)
    pairs       : list of lists
                Each sublist contains the index of a residue composing the pair
                e.g., [[0, 1], [2, 3]] for the pairs 0-1 and 2-3
    cutoff      : float
                Cutoff value. Pairs within the cutoff are included, others are
                excluded.
    Returns
    -------
    retain_list : list of bool
                Contains True if the pair is within the cutoff, False otherwise.
    """
    retain_list = []
    iterator = pbar(
        pairs,
        desc=": Evaluating distances",
        ncols=79,
    )
    for pair in iterator:
        iterator.set_postfix(pair=f"{residue_ids[pair[0]]}.{residue_ids[pair[1]]}")
        coords1 = coords[pair[0]]
        coords2 = coords[pair[1]]
        dist = _distance_geometric_center(coords1, coords2)
        # Decide on the basis of the mean distance
        # if the pair has to be retained or not
        criterion = np.min(dist)
        if criterion < cutoff:
            retain = True
        else:
            retain = False
        retain_list.append(retain)
    return retain_list


def apply_distance_cutoff(coords, residue_ids, pairs, cutoff):
    """
    Apply a cutoff based on the distance between molecule pairs.
    Arguments
    ---------
    coords      : list of ndarray, (num_frames, num_atoms, 3)
                List of atomic coordinates, each entry corresponds to a molecule
    residue_ids : list of str
                Indeces/names of the residue_ids (used for printing only)
    pairs       : list of lists
                Each sublist contains the index of a residue composing the pair
                e.g., [[0, 1], [2, 3]] for the pairs 0-1 and 2-3
    cutoff      : float
                Cutoff value. Pairs within the cutoff are included, others are
                excluded.
    Returns
    -------
    pairs       : list of lists
                Each sublist contains the index of a residue composing the pair,
                only for pairs within the cutoff
                e.g., [[0, 1]] if only residues 0-1 are within the cutoff.
    """
    retain_list = _compute_retain_list(coords, residue_ids, pairs, cutoff)
    pairs = [p for p, retain in zip(pairs, retain_list) if retain == True]  # noqa: E712
    return pairs


def build_connectivity_matrix(topology, count_as="python"):
    """
    Build the connectivity matrix given a topology
    Arguments
    ---------
    topology     : pytraj.Topology
                 Trajectory topology
    count_as     : str
                 If 'fortran', counts starting from 1
                 If 'python', counts starting from 0
    Returns
    -------
    connectivity : ndarray, (num_atoms, max_connections)
                 Connectivity matrix, storing the indeces
                 of the atoms connected to each other by a bond
    """
    c = _count_as(count_as)
    connectivity = []
    for atom in topology.atoms:
        connectivity.append(atom.bonded_indices() + c)
    connectivity = _pad_list_of_lists(connectivity, fillvalue=-1)
    return connectivity


# =============================================================================
# Coulomb coupling functions
# =============================================================================


def _coulomb_coupling(coord_pairs, charge_pairs):
    """
    Compute the Coulomb coupling between two
    distributions of point charges.
    Arguments
    ---------
    coord_pairs  : list of two ndarray, (num_samples, num_atoms, 3)
                 cartesian coordinates.
    charge_pairs : list of two ndarray, (num_samples, num_atoms)
                 point charges.
    Returns
    -------
    V            : ndarray, (num_frames,)
                 Coulomb coupling.
    """
    n_atoms1 = charge_pairs[0].shape[1]
    n_atoms2 = charge_pairs[1].shape[1]
    pairs = [
        (p[0], p[1] + n_atoms1) for p in iterproduct(range(n_atoms1), range(n_atoms2))
    ]
    xyz_full = np.concatenate(coord_pairs, axis=1)
    charge_full = np.concatenate(charge_pairs, axis=1)
    rinv = np.diff(xyz_full[:, pairs, :], axis=2)[:, :, 0] ** 2
    rinv = pow(rinv.sum(axis=2), -0.5)
    qq = charge_full[:, pairs].prod(axis=2)
    V = np.sum(qq * rinv, axis=1)
    return V


def coulomb_coupling(coords, charges, residue_ids, pairs, cutoff):
    """
    Compute the Coulomb coupling between molecule pairs for a density
    projected onto point charges.
    Arguments
    ---------
    coords      : list of ndarray, (num_samples, num_atoms, 3)
                Atomic coordinates.
    charges     : list of ndarray, (num_samples, num_atoms,)
                Point charges.
    residue_ids : list of str
                Indices/names of the molecules.
    pairs       : list of lists
                Each sublist contains the indeces of the residues forming the pair
                e.g., [[0, 1], [2, 3]] for pairs 0-1 and 2-3.
    cutoff      : float
                Cutoff value. The coupling is computed for pairs within the cutoff.
    Returns
    -------
    couplings   : list of ndarray, (num_samples,)
                Coulomb couplings.
    residue_ids : list of lists
                Each sublist contains the residue_ids for which the coupling has been computed.
    """
    # Apply a distance cutoff on the list of molecules
    molecule_pairs = apply_distance_cutoff(coords, residue_ids, pairs, cutoff)
    if len(molecule_pairs) < 1:
        # No coupling to be computed
        return (None, None)
    # List of pairs of coordinates, each of shape (num_samples, num_atoms, 3)
    coord_pairs = [(coords[pair[0]], coords[pair[1]]) for pair in molecule_pairs]
    # List of pairs of charges, each of shape (num_samples, num_atoms)
    charge_pairs = [(charges[pair[0]], charges[pair[1]]) for pair in molecule_pairs]
    couplings = []
    iterator = pbar(
        zip(coord_pairs, charge_pairs, molecule_pairs),
        total=len(coord_pairs),
        desc=": Computing couplings",
        ncols=79,
    )
    for coords, charges, pair in iterator:
        iterator.set_postfix(pair=f"{residue_ids[pair[0]]}.{residue_ids[pair[1]]}")
        coup = _coulomb_coupling(coord_pairs=coords, charge_pairs=charges)
        # Append and convert to cm^{-1}
        couplings.append(coup * HARTREESQ_NM2CM_1 * 10.0)
    # We also have to tell back what are the residues for which we have
    # computed the coupling, since we apply a threshold
    residue_ids = [
        (residue_ids[pair[0]], residue_ids[pair[1]]) for pair in molecule_pairs
    ]
    return couplings, residue_ids


def compute_coulomb_couplings(coords, charges, residue_ids, cutoff):
    """
    Compute the Coulomb interaction between all
    unique pairs of molecules.
    Arguments
    ---------
    coords      : list of ndarray, (num_samples, num_atoms, 3)
                Atomic coordinates.
    charges     : list of ndarray, (num_samples, num_atoms,)
                Point charges.
    residue_ids : list of str
                Indices/names of the molecules.
    cutoff      : float
                Cutoff value. The coupling is computed for pairs within the cutoff.
    Return
    ------
    couplings      : list of ndarray, (num_frames,)
                   Coulomb couplings.
    residue_ids    : list of tuples, (2,)
                   list of tuples, each storing the residue IDs of the
                   two molecule for which the coupling is computed.
    """
    num_molecules = len(coords)
    pairs = [
        (i, j) for i in range(num_molecules - 1) for j in range(i + 1, num_molecules)
    ]
    return coulomb_coupling(coords, charges, residue_ids, pairs, cutoff)


# =============================================================================
# Saving functions
# =============================================================================


def backup_file(file, n):
    """
    Rename "file" by appending ".bck{:d}".format(n).
    If a file with that name already exists, n is increased.
    Arguments
    ---------
    file      : str
              Filename
    n         : int
              Backup index
    """
    newfile = file + ".bck{:d}".format(n)
    if os.path.isfile(newfile):
        backup_file(file, n + 1)
    else:
        os.rename(file, newfile)


def create_hdf5_outfile(outfile):
    """
    Create an empty HDF5 file.
    If a file with the same name exists, it is renamed in order
    to prevent overwriting.
    Arguments
    ---------
    outfile   : str
              Filename
    """
    # If the file is already there, create a backup copy
    if os.path.isfile(outfile):
        backup_file(outfile, 0)
    with h5py.File(outfile, "w"):
        pass


def save_n_frames(n_frames, outfile):
    """
    Save the number of frames to an existing HDF5 file.
    Arguments
    ---------
    n_frames  : int
              Number of frames
    outfile   : str
              Filename
    """
    with h5py.File(outfile, "a") as hf:
        hf.create_dataset("n_frames", data=n_frames)


def save_residue_ids(residue_ids, outfile):
    """
    Save the residue ids to an existing HDF5 file.
    Arguments
    ---------
    residue_ids  : list of str
                 Residue IDs
    outfile      : str
                 Filename
    """
    with h5py.File(outfile, "a") as hf:
        hf.create_dataset("residue_ids", data=np.asarray(residue_ids).astype("S"))


def save_coords(coords, residue_ids, outfile):
    """
    Save the atomic coordinates to an existing HDF5 file.
    Arguments
    ---------
    coords      : list of ndarray, (num_samples, num_atoms, 3)
                List of atomic coordinates
    residue_ids : list of str
                Residue IDs
    outfile     : str
                Filename
    """
    group = "xyz"
    with h5py.File(outfile, "a") as hf:
        if group not in hf.keys():
            hf.create_group(group)
        for r, c in zip(residue_ids, coords):
            hf[group].create_dataset(r, data=c)


def save_atom_numbers(atnums, residue_ids, outfile):
    """
    Save the atomic numbers to an existing HDF5 file.
    Arguments
    ---------
    atnums      : list of ndarray, (num_atoms,)
                Atomic numbers
    residue_ids : list of str
                Residue IDs
    outfile     : str
                Filename
    """
    group = "atnums"
    with h5py.File(outfile, "a") as hf:
        if group not in hf.keys():
            hf.create_group(group)
        for r, a in zip(residue_ids, atnums):
            hf[group].create_dataset(r, data=a)


def save_tresp_charges(charges, residue_ids, kind, outfile):
    """
    Save the point charges to an existing HDF5 file.
    Arguments
    ---------
    charges     : list of ndarray, (num_samples, num_atoms)
                Point charges
    residue_ids : list of str
                Residue IDs
    kind        : str
                Kind of point charges (e.g., vac, env, env_pol)
    outfile     : str
                Filename
    """
    group = f"tresp/{kind}"
    with h5py.File(outfile, "a") as hf:
        if group not in hf.keys():
            hf.create_group(group)
        for r, c in zip(residue_ids, charges):
            hf[group].create_dataset(r, data=c)


def save_dipoles(dipoles, residue_ids, kind, outfile):
    """
    Save the dipoles to an existing HDF5 file.
    Arguments
    ---------
    dipoles     : list of ndarray, (num_samples, 3)
                Dipoles
    residue_ids : list of str
                Residue IDs
    kind        : str
                Kind of point charges (e.g., vac, env, env_pol)
    outfile     : str
                Filename
    """
    group = f"trdipole/{kind}"
    with h5py.File(outfile, "a") as hf:
        if group not in hf.keys():
            hf.create_group(group)
        for r, d in zip(residue_ids, dipoles):
            hf[group].create_dataset(r, data=d)


def save_coulomb_couplings(couplings, pairs_ids, kind, outfile):
    """
    Save the Coulomb couplings to an existing HDF5 file.
    Arguments
    ---------
    couplings : list of ndarray, (num_samples,)
              Coulomb couplings
    pairs_ids : list of lists
              Residue IDs composing each pair
    kind      : str
              Kind of coupling (e.g., vac, env, env_pol)
    outfile   : str
              Filename
    """
    group = f"coup/{kind}"
    pairs = [f"{p[0]}_{p[1]}" for p in pairs_ids]
    with h5py.File(outfile, "a") as hf:
        if group not in hf.keys():
            hf.create_group(group)
        for p, c in zip(pairs, couplings):
            hf[group].create_dataset(p, data=c)


def save_site_energies(mean_energies, var_energies, residue_ids, kind, outfile):
    """
    Save the site energies to an existing HDF5 file.
    Arguments
    ---------
    mean_energies : list of ndarray, (num_samples,)
                  Site energies (mean value)
    var_energies  : list of ndarray, (num_samples,)
                  Site energies (variance)
    residue_ids   : list of str
                  Residue IDs
    kind          : str
                  Kind of site energy (e.g., vac, env, env_pol)
    outfile       : str
                  Filename
    """
    with h5py.File(outfile, "a") as hf:
        for r, me, ve in zip(residue_ids, mean_energies, var_energies):
            group = f"siten/{kind}/{r}"
            if group not in hf:
                hf.create_group(group)
            try:
                me = me.numpy()
                ve = ve.numpy()
            except AttributeError:
                pass
            hf[group].create_dataset("mean", data=me)
            hf[group].create_dataset("var", data=ve)


# =============================================================================
# Reading functions
# =============================================================================


def load_n_frames(outfile):
    """
    Load the number of frames from a HDF5 file.
    Arguments
    ---------
    outfile   : str
              Filename
    Returns
    -------
    n_frames  : int
              Number of frames
    """
    with h5py.File(outfile, "r") as hf:
        n_frames = np.asarray(hf.get("n_frames"))
    return n_frames


def load_residue_ids(outfile):
    """
    Load the residue IDs from a HDF5 file.
    Arguments
    ---------
    outfile   : str
              Filename
    Returns
    -------
    residue_ids : list of str
                Residue IDs
    """
    with h5py.File(outfile, "r") as hf:
        residue_ids = np.asarray(hf.get("residue_ids").asstr())
    return residue_ids


def load_coords(outfile, frames=None):
    """
    Load the atomic coordinates from a HDF5 file.
    Arguments
    ---------
    outfile   : str
              Filename
    frames    : bool or ndarray
              Frames to load
    Returns
    -------
    coords    : dict
              Dictionary with residue id (key) and coordinates (value)
    """
    coords = dict()
    group = "xyz"
    with h5py.File(outfile, "r") as hf:
        for resid in hf[group].keys():
            coords[resid] = np.asarray(hf[group].get(resid))
            if frames is not None:
                coords[resid] = coords[resid][frames]
    return coords


def load_atom_numbers(outfile):
    """
    Load the atomic numbers from a HDF5 file.
    Arguments
    ---------
    outfile   : str
              Filename
    Returns
    -------
    atnums    : dict
              Dictionary with residue ID (key) and atomic numbers (value)
    """
    atnums = dict()
    group = "atnums"
    with h5py.File(outfile, "r") as hf:
        for resid in hf[group].keys():
            atnums[resid] = np.asarray(hf[group].get(resid))
    return atnums


def load_tresp_charges(outfile, kind, frames=None):
    """
    Load the point charges from a HDF5 file.
    Arguments
    ---------
    outfile   : str
              Filename
    kind      : str
              Kind of charges to load
              (e.g., vac, env, env_pol)
    frames    : None or ndarray
              Frames to load
    Returns
    -------
    tresp     : dict
              Dictionary with residue ID (key) and charges (value)
    """
    tresp = dict()
    group = f"tresp/{kind}"
    with h5py.File(outfile, "r") as hf:
        for resid in hf[group].keys():
            tresp[resid] = np.asarray(hf[group].get(resid))
            if frames is not None:
                tresp[resid] = tresp[resid][frames]
    return tresp


def load_dipoles(outfile, kind, frames=None):
    """
    Load the transition dipoles from a HDF5 file.
    Transition dipoles are in AU.
    Arguments
    ---------
    outfile   : str
              Filename
    kind      : str
              Kind of dipoles to load
              (e.g., vac, env, env_pol)
    frames    : None or ndarray
              Frames to load
    Returns
    -------
    dipoles   : dict
              Dictionary with residue ID (key) and dipoles (value)
    """
    dipoles = dict()
    group = f"trdipole/{kind}"
    with h5py.File(outfile, "r") as hf:
        for resid in hf[group].keys():
            dipoles[resid] = np.asarray(hf[group].get(resid))
            if frames is not None:
                dipoles[resid] = dipoles[resid][frames]
    return dipoles


def load_couplings(outfile, kind, units="cm_1", frames=None):
    """
    Load the couplings from a HDF5 file.
    Arguments
    ---------
    outfile   : str
              Filename
    kind      : str
              Kind of coupling to load
              (e.g., vac, env, env_pol)
    units     : str
              Units (cm_1 or eV)
    frames    : None or ndarray
              Frames to load
    Returns
    -------
    coups     : dict
              Dictionary with residue ID (key) and coupling (value)
    """
    coups = dict()
    group = f"coup/{kind}"
    with h5py.File(outfile, "r") as hf:
        for pair in hf[group].keys():
            subgroup = group + "/" + pair
            coups[pair] = np.asarray(hf[subgroup]).reshape(-1)
            if frames is not None:
                coups[pair] = coups[pair][frames]
            if units == "eV":
                coups[pair] /= EV2CM
            elif units == "cm_1":
                pass
            else:
                raise RuntimeError(f"Units={units} not recognized.")
    return coups


def load_site_energies(outfile, kind, units="cm_1", frames=None):
    """
    Load the site energies from a HDF5 file.
    Arguments
    ---------
    outfile   : str
              Filename
    kind      : str
              Kind of site energies to load
              (e.g., vac, env, env_pol)
    units     : str
              Units (cm_1 or eV)
    frames    : None or ndarray
              Frames to load
    Returns
    -------
    sites     : dict
              Dictionary with residue ID (key) and site energy (value)
    """
    sites = defaultdict(dict)
    group = f"siten/{kind}"
    with h5py.File(outfile, "r") as hf:
        for resid in hf[group].keys():
            subgroup = group + "/" + resid
            sites[resid]["mean"] = np.asarray(hf[subgroup].get("mean")).reshape(-1)
            sites[resid]["var"] = np.asarray(hf[subgroup].get("var")).reshape(-1)
            if frames is not None:
                sites[resid]["mean"] = sites[resid]["mean"][frames]
                sites[resid]["var"] = sites[resid]["var"][frames]
            if units == "eV":
                pass
            elif units == "cm_1":
                sites[resid]["mean"] *= EV2CM
                sites[resid]["var"] *= EV2CM
            else:
                raise RuntimeError(f"Units={units} not recognized.")
    return sites


# =============================================================================
# Miscellaneous/helper functions
# =============================================================================


def _count_as(language):
    if language.upper() == "FORTRAN":
        return 1
    elif language.upper() == "PYTHON":
        return 0
    else:
        raise RuntimeError(
            "Language {language} not supported. Specify either Fortran or Python."
        )


def _pad_list_of_lists(list_of_lists, fillvalue=0):
    """
    Return a ndarray from a list of lists of various lengths, with lists
    padded with `fillvalue` if they are shorter than the maximum length
    """
    return np.asarray(list(zip_longest(*list_of_lists, fillvalue=fillvalue))).T


def read_molecule_types(topology, residue_ids):
    """
    Get the type (resname) of each molecule specified
    by its residue ID
    Arguments
    ---------
    topology    : pytraj.Topology
                Trajectory topology
    residue_ids : list of str
                Residue IDs
    Returns
    -------
    types       : list of str
                Molecule types
    """
    types = []
    for residue_id in residue_ids:
        type = [r.name for r in topology[":" + str(residue_id)].residues][0]
        types.append(type)
    return types


def make_pair_mask(mask1, mask2):
    """
    Given two AMBER-like masks, make a masks
    that selects both
    """
    return "(" + mask1 + ")|(" + mask2 + ")"


def select_molecules(residue_ids, list_of_resids):
    """
    Select molecules from `molecules` given a list
    of lists of residue IDs.
    Arguments
    ---------
    molecules      : list of excipy.Molecule objects
                   List of molecules to be selected
    list_of_resids : list of lists
                   List of residue IDs of the selected molecules.
    Residues
    --------
    """
    selected_molecules = []
    for resids in list_of_resids:
        selected = []
        for resid in resids:
            for i, residue_id in enumerate(residue_ids):
                if residue_id == resid:
                    selected.append(i)
        selected_molecules.append(selected)
    return selected_molecules


def rescale_tresp(charges, scalings):
    """
    Rescale the charges with the provided scalings
    Arguments
    ---------
    charges    : list of ndarray, (num_samples, num_atoms)
               Point charges
    scalings   : list of float
               Scaling factors
    Returns
    -------
    scaled_tresp : list of ndarray, (num_frames, num_atoms)
                 Scaled TrEsp charges.
    """
    scaled_tresp = []
    for charge, scaling in zip(charges, scalings):
        charge *= scaling
        scaled_tresp.append(charge)
    return scaled_tresp


def get_dipoles(coords, charges):
    """
    Compute the transition dipoles given the coordinates and the charges
    Transition dipoles are computed in AU.
    Arguments
    ---------
    coords    : list of ndarray, (num_samples, num_atoms, 3)
              Atomic coordinates
    charges   : list of ndarray, (num_samples, num_atoms)
              Point charges
    Returns
    -------
    dipoles   : list of ndarray, (num_samples, 3)
              Dipoles
    """
    dipoles = []
    # Here coordinates are in Angstrom
    for xyz, q in zip(coords, charges):
        dipole = np.sum(xyz * q[:, :, None], axis=1) * ANG2BOHR
        dipoles.append(dipole)
    return dipoles


# =============================================================================
# EXAT-related functions
# =============================================================================


def _load_exat_quantities(
    infile,
    frames=None,
    sites_kind="vac",
    coups_kind="vac",
    dipoles_kind="vac",
    ene_units="cm_1",
):
    """
    Load quantities necessary to construct an exat.ExcSystem from an excipy HDF5 output file.
    Arguments
    ---------
    frames        : None or ndarray
                  Frames to load
    sites_kind    : str
                  Kind of site energy to load (vac, env, env_pol)
    coups_kind    : str
                  Kind of coupling to load (vac, env, env_pol)
    dipoles_kind  : str
                  Kind of dipoles to load (vac, env, env_pol)
    ene_units     : str
                  Energy units (cm_1 or eV)
    Returns
    -------
    d             : dict
                  Dictionary with quantities necessary to construct an ExcSystem.
                  Each quantity is loaded "per residue".
    """

    # Residue IDs
    residue_ids = load_residue_ids(infile)
    # EXAT ChromList
    # (we are able to estimate only one transition per chromophore)
    chromlist = {r: [1] for r in residue_ids}
    # Coordinates of each chromophore
    coords = load_coords(infile, frames=frames)
    # Geometric center of each chromophore
    centers = {r: np.mean(c, axis=1) for r, c in coords.items()}
    # Atom numbers for each chromophore
    atnums = load_atom_numbers(infile)
    # Total number of atoms for each chromophore
    natoms = {k: len(v) for k, v in atnums.items()}
    # Site energies (cm^-1)
    sites = load_site_energies(infile, kind=sites_kind, units=ene_units, frames=frames)
    sites = {k: v["mean"] for k, v in sites.items()}
    # Couplings (cm^-1)
    coups = load_couplings(infile, kind=coups_kind, units=ene_units, frames=frames)
    # Transition Dipoles
    dipoles = load_dipoles(infile, kind=dipoles_kind, frames=frames)
    return dict(
        ChromList=chromlist,
        xyz=coords,
        Cent=centers,
        anum=atnums,
        NAtom=natoms,
        site=sites,
        coup=coups,
        DipoLen=dipoles,
    )


def load_as_exat(
    infile,
    frames=None,
    sites_kind="vac",
    coups_kind="vac",
    dipoles_kind="vac",
    ene_units="cm_1",
):
    """
    Load the excipy output in a format that can be employed easily to construct
    an exat.ExcSystem.
    Each exat.ExcSystem field (ChromList, xyz, Cent, ...) is loaded with the corresponding
    name.
    Iterating over this fields corresponds to an iteration over the trajectory frames.
    Arguments
    ---------
    frames        : None or ndarray
                  Frames to load
    sites_kind    : str
                  Kind of site energy to load (vac, env, env_pol)
    coups_kind    : str
                  Kind of coupling to load (vac, env, env_pol)
    dipoles_kind  : str
                  Kind of dipoles to load (vac, env, env_pol)
    ene_units     : str
                  Energy units (cm_1 or eV)
    Returns
    -------
    eq        : dict
              Dictionary with exat.ExcSystem fields as keys
              Iterating over each value corresponds to iterating over
              the trajectory frames.
    """

    # Load EXAT quantities from the excipy output.
    # Quantities are loaded *per residue*
    eq = _load_exat_quantities(
        infile,
        frames=frames,
        sites_kind=sites_kind,
        coups_kind=coups_kind,
        dipoles_kind=dipoles_kind,
        ene_units=ene_units,
    )
    residue_ids = list(eq["ChromList"].keys())
    n_frames = len(eq["xyz"][residue_ids[0]])

    # Conversion to *per frame* objects
    # list of EXAT-like, shape (n_chrom*n_atoms, 3)
    eq["xyz"] = [
        np.concatenate(c, axis=0) for c in zip(*[eq["xyz"][r] for r in residue_ids])
    ]
    # list of EXAT-like, shape: (n_chrom, 3)
    eq["Cent"] = [np.asarray(c) for c in zip(*[eq["Cent"][r] for r in residue_ids])]
    # EXAT-like, shape: (n_chrom*n_atoms)
    eq["anum"] = np.concatenate([eq["anum"][r] for r in residue_ids])
    # EXAT-like, shape: (n_chrom,)
    eq["NAtom"] = np.asarray([eq["NAtom"][r] for r in residue_ids])
    # list of EXAT-like, shape (n_chrom,)
    eq["site"] = [np.asarray(c) for c in zip(*[eq["site"][r] for r in residue_ids])]
    # list of EXAT-like, shape (n_chrom * (n_chrom - 1) / 2, )
    # We have to account for missing couplings (for which we put an array of zeros)
    # Create a list of all the pairs (with no repetitions)
    pairs = []
    for i in range(len(residue_ids) - 1):
        for j in range(i + 1, len(residue_ids)):
            pairs.append(residue_ids[i] + "_" + residue_ids[j])
    for pair in pairs:
        if pair not in eq["coup"].keys():
            eq["coup"][pair] = np.zeros(n_frames)
        else:
            pass
    eq["coup"] = [np.asarray(c) for c in zip(*[eq["coup"][p] for p in pairs])]
    # list of EXAT-like, shape (n_chrom, 3)
    eq["DipoLen"] = [
        np.asarray(c) for c in zip(*[eq["DipoLen"][r] for r in residue_ids])
    ]
    # We do not provide DipoVel and Mag: we fill with zeros
    eq["DipoVel"] = [np.zeros(c.shape) for c in eq["DipoLen"]]
    eq["Mag"] = [np.zeros(c.shape) for c in eq["DipoLen"]]

    return eq


def _build_hamiltonian(site, coup):
    """
    Construct an exciton Hamiltonian from site energies and couplings.
    """
    n = site.shape[0]
    H = np.zeros((n, n))
    H[np.triu_indices(n, 1)] = coup
    H += H.T
    np.fill_diagonal(H, site)
    return H


def dump_exat_files(exat_quantities, outname="exat"):
    """
    Dump the loaded exat quantities as a NumPy`s .npz file
    that can be parsed directly with exat.
    Arguments
    ---------
    exat_quantities  : dict
                     Dictionary with exat.ExcSystem fields as keys
                     Iterating over the values has to correspond to an
                     iteration over the trajectory frames.
    outname          : str
                     Output file name (prefix)
    """
    # check: site, coup, xyz, Cent, DipoLen, DipoVel, Mag
    # should all be of "n_frames" length
    eq = exat_quantities
    n_frames = len(eq["site"])
    check = ["site", "coup", "xyz", "Cent", "DipoLen", "DipoVel", "Mag"]
    for entry in check:
        if len(eq[entry]) != n_frames:
            raise RuntimeError("Wrong number of frames in exat_quantities")

    iterator = pbar(
        range(n_frames),
        desc=": Dumping EXAT files",
        ncols=79,
    )

    for frame in iterator:
        np.savez(
            "{:s}.{:05d}.npz".format(outname, frame + 1),
            **dict(
                ChromList=eq["ChromList"],
                xyz=eq["xyz"][frame],
                anum=eq["anum"],
                NAtom=eq["NAtom"],
                site=eq["site"][frame],
                coup=eq["coup"][frame],
                Cent=eq["Cent"][frame],
                DipoLen=eq["DipoLen"][frame],
                DipoVel=eq["DipoVel"][frame],
                Mag=eq["Mag"][frame],
                Kappa=None,
                #     H=_build_hamiltonian(site=eq['site'][frame], coup=eq['coup'][frame]),
                H=None,  # None in order to save memory.
            ),
        )


# =============================================================================
# Colors
# =============================================================================


class Colors:
    bold = "\033[1m"
    null = "\033[0m"
    gray = "\033[90m"
    red = "\033[91m"
    green = "\033[92m"
    orange = "\033[93m"
    blue = "\033[94m"
    magenta = "\033[95m"


# =============================================================================
# Statistics
# =============================================================================


def _block_average(array, n_blocks):
    """
    Block-averaging over the leading dimension of an array.
    If the array cannot be equally split into sub-arrays, the last n
    elements of the array are ignored.
    """
    rem = array.shape[0] % n_blocks
    if rem != 0:
        array = array[:-rem]
    subarrays = np.split(array, n_blocks, axis=0)
    subarrays = [np.mean(sa, axis=0) for sa in subarrays]
    return subarrays


def block_average(arrays, n_blocks):
    """
    Block-averaging a list of arrays over the leading dimension of each array.
    If one array cannot be equally split into sub-arrays, the last n elements
    of the array are ignored.
    """
    return [_block_average(a, n_blocks) for a in arrays]
