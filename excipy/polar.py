import numpy as np
from scipy.sparse.linalg import LinearOperator, cg
from scipy.spatial.distance import cdist
import pytraj as pt
from tqdm import tqdm
from .tmu import tmu as tmu_fortran
from .util import ANG2BOHR, HARTREE2CM_1
from .util import (
    make_pair_mask,
    build_connectivity_matrix,
    atomnum2atomsym,
    _count_as,
    _pad_list_of_lists,
    pbar,
)


# =============================================================================
# Polarization models
# for now only WangAL is supported.
# =============================================================================

WANGAL = {
    "B": 2.24204,  # Boron (J Phys Chem A, 103, 2141)
    "Mg": 0.1200,  # Magnesium (???)
    "C1": 1.3916,  # sp1 Carbon
    "C2": 1.2955,  # sp2 Carbon
    "C3": 0.9399,  # sp3 Carbon
    "H": 0.4255,  #
    "NO": 1.4824,  # Nitro Nitrogen
    "N": 0.9603,  # Other Nitrogen
    "O2": 0.6049,  # sp2 Oxygen
    "O3": 0.6148,  # sp3 Oxygen
    "F": 0.4839,  #
    "Cl": 2.3707,  #
    "Br": 3.5016,  #
    "I": 5.5788,  #
    "S4": 2.3149,  # S in sulfone
    "S": 3.1686,  # Other S
    "P": 1.7927,  #
    "Zn": 0.2600,  # Zinc (II) from AMOEBA (JCTC, 6, 2059)
}

WANGAL_FACTOR = 2.5874


# =============================================================================
# Utilities to deal with QM and MM subsystems and polarizable stuff
# =============================================================================


def compute_environment_mask(topology, mask1, mask2):
    """
    Given the masks of two pigments, not considered part of
    the environment, get the mask of the environment atoms.
    Arguments
    ---------
    topology  : pytraj.Topology
              Trajectory topology
    mask1     : str
              Mask of the first pigment
    mask2     : str
              Mask of the second pigment
    Returns
    -------
    mask     : ndarray, (num_atoms)
             Array of True (environment atom) or
             False (pair atom)
    """
    num_atoms = topology.n_atoms
    pair_mask = make_pair_mask(mask1, mask2)
    env_mask = np.full(num_atoms, True)
    env_mask[topology.select(pair_mask)] = False
    return env_mask


def _water_polarizability(atnum):
    """
    Get the atomic polarizability for a water atom.
    NOTE: currently not used but kept here for reference.
    """
    if atnum == 8:
        return 1.49073
    elif atnum == 1:
        return 0.0
    else:
        raise RuntimeError("Water atom is neither oxygen nor hydrogen.")


def _nitro_sulfone_symbol(topology, atom):
    """
    Get the atomic symbol for a nitro or sulfone atom
    """
    atnum = atom.atomic_number
    sym = atomnum2atomsym[atnum]
    bonded_atoms = _get_bonded_atoms(atom, topology)
    bonded_oxy_atoms = [a for a in bonded_atoms if a.atomic_number == 8]
    num_oxygens = len(bonded_oxy_atoms)
    if num_oxygens == 2:
        # We need two oxygens with valence one
        oxy_n_bonds = [a.n_bonds for a in bonded_oxy_atoms]
        if oxy_n_bonds == [1, 1]:
            # Real nitro/sulfone
            if sym == "N":
                return "NO"
            else:
                return "S4"
        else:
            return sym
    else:
        return sym


def _compute_polarizabilities(topology, poldict=WANGAL):
    """
    Assign the atomic polarizability for each atom in the topology.
    Arguments
    ---------
    topology  : pytraj.Topology
              Trajectory topology
    poldict   : dict
              Dictionary mapping atom symbols to polarizabilities
    Returns
    -------
    polarizabilities : ndarray, (num_atoms,)
                     Atomic polarizabilities
    """
    atoms = topology.atoms
    polarizabilities = []
    for atom in atoms:
        atnum = atom.atomic_number
        resname = atom.resname
        # If uncommenting this if-else, the next "if"
        # should become an "elif"
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        ## Waters get their own parameters
        # if resname in ['WAT', 'WCR']:
        #    pol = _water_polarizability(atnum)
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Simple case, no ambiguities due to
        # hybridization of the atom:
        #            H  B  F  Mg  P   Cl  Br  I
        if atnum in [1, 5, 9, 12, 15, 17, 35, 53]:
            sym = atomnum2atomsym[atnum]
            pol = poldict[sym]
        # Cases with more than one type
        else:
            if atnum == 6:  # C
                sym = "C{:1d}".format(atom.n_bonds - 1)
            elif atnum == 8:  # O
                sym = "O{:1d}".format(atom.n_bonds + 1)
            # Distinguish nitro and sulfone
            elif atnum in [7, 16]:  # N, S
                sym = _nitro_sulfone_symbol(topology, atom)
            # Polarizability not present for this atom
            else:
                sym = None
            if sym is not None:
                pol = poldict[sym]
            else:
                pol = 0.0
        polarizabilities.append(pol)
    return np.asarray(polarizabilities)


def compute_polarizabilities(topology, poldict=WANGAL):
    """
    Assign the atomic polarizability for each atom in the topology.
    Arguments
    ---------
    topology  : pytraj.Topology
              Trajectory topology
    poldict   : dict
              Dictionary mapping atom symbols to polarizabilities
    Returns
    -------
    polarizabilities : ndarray, (num_atoms,)
                     Atomic polarizabilities
    """
    polarizabilities = _compute_polarizabilities(topology, poldict)
    polarizabilities *= ANG2BOHR**3
    return polarizabilities


def _get_bonded_atoms(atom, topology):
    """
    Get the pytraj.Atom objects bonded to a pytraj.Atom.
    """
    return [topology.atom(i) for i in atom.bonded_indices()]


def _angstrom2bohr(*arrays):
    """
    Convert from angstrom to bohr units an arbitrary number of arrays.
    """
    return (a * ANG2BOHR for a in arrays)


# =============================================================================
# MMPol Manipulation Functions
# =============================================================================


def _polarization_cutoff(
    coords, env_mask, pol_threshold, full_connect, polarizabilities
):
    """
    Select the polarization part based on the `mmpol_threshold`.
    Arguments
    ---------
    coords           : ndarray, (num_atoms, 3)
                     Cartesian coordinates of the full system for a single frame.
    env_mask         : ndarray, (num_atoms,)
                     Mask that selects the environment atoms.
    pol_threshold    : float
                     Threshold for the polarization part.
    full_connect     : ndarray, (num_atoms, max_connections)
                     full connectivity matrix
    polarizabilities : ndarray, (num_atoms,)
                     Atomic polarizabilities
    """
    # Total number of atoms (env+pair)
    num_atoms = len(env_mask)
    env_idx = np.where(env_mask)[0]
    # Coordinates of the environment and the chromophore pair
    env_coords = coords[env_mask]
    pair_coords = coords[~env_mask]
    # Compute the distances between each atom of the pigment
    # pair and every other atom of the environment
    # dist is of shape (num_env_atoms, num_pair_atoms)
    dist = cdist(env_coords, pair_coords)
    # For each environment atom, find the minimum distance
    # from the pair coordinates
    # min_dist is of shape (num_env_atoms,)
    min_dist = np.min(dist, axis=1)
    # Keep atoms within the cutoff
    keep_mask = min_dist <= pol_threshold
    cut_mask = np.zeros(num_atoms, dtype=bool)
    cut_mask[env_idx[keep_mask]] = True
    # Exclude atoms with polarizability zero
    cut_mask[polarizabilities == 0] = False
    return cut_mask


def _map_polarizable_atoms(cut_mask, count_as="fortran"):
    """
    Builds a map from an array of the indices of all atoms
    to an array with the indices of the polarizable atoms.
    Arguments
    ---------
    cut_mask    : ndarray, (num_atoms,)
                Mask of bool, with True for polarizable atoms
                within the polarization threshold and False otherwise
    count_as    : str
                Either 'fortran' (starts counting from 1)
                or 'python' (starts counting from 0)
    """
    num_atoms = len(cut_mask)
    num_pol_atoms = sum(cut_mask)
    # Create a map from all atoms to the polarizable atoms
    c = _count_as(count_as)
    pol_idx = np.where(cut_mask)[0]
    pol_map = np.zeros(num_atoms + c, dtype=int)
    pol_map[pol_idx + c] = np.arange(num_pol_atoms, dtype=int) + c
    return pol_map, num_pol_atoms, pol_idx


def _build_pol_neighbor_list(full_connect, cut_mask):
    """
    Build the polarizable neighbor list needed to compute the exclusion
    list in the Fortran routine.
    """
    pol_map, num_pol_atoms, pol_idx = _map_polarizable_atoms(
        cut_mask, count_as="fortran"
    )
    m, n = full_connect.shape
    pol_connect = np.zeros((num_pol_atoms, n), dtype=int)
    for i, idx in enumerate(pol_idx):
        pol_connect[i] = pol_map[full_connect[idx]] - 1
    # Nearest neighbor list
    nn_list = []
    for i in range(num_pol_atoms):
        n12 = pol_connect[i]
        n12 = n12[n12 >= 0].tolist()
        n13 = pol_connect[n12]
        n13 = n13[n13 != i].tolist()
        nn_list.append(n12 + n13)
    nn_list = _pad_list_of_lists(nn_list, fillvalue=-1) + 1
    return nn_list


def electric_field(target_coords, source_coords, source_charges):
    """
    Computes the electric fields at points `target_coords`
    produced by a point charge distribution of point charges
    `source_charges` localized at `source_coords`.
    Arguments
    ---------
    target_coords  : ndarray, (num_targets, 3)
                   Target coordinates
    source_coords  : ndarray, (num_sources, 3)
                   Source coordinates
    source_charges : ndarray, (num_sources,)
                   Source charges
    Returns
    -------
    E              : ndarray, (num_targets, 3)
                   Electric field at the target coordinates
    """
    dist = target_coords[:, None, :] - source_coords[None, :, :]
    dist2 = np.sum(dist**2, axis=2)
    dist3 = dist2 * dist2**0.5
    ddist3 = dist / dist3[:, :, None]
    E = np.sum(ddist3 * source_charges[None, :, None], axis=1)
    return E


# =============================================================================
# Linear System Solvers Interface
# =============================================================================


class TmuOperator(LinearOperator):
    """
    Interface operator for computing T*mu when solving linear systems
    using SciPy's classes. This interface provides a method to compute
    T*mu, i.e., a linear system is solvable without computing the full T.
    Can be used when mu has shape (N,3) and we want to pass some additional arguments.
    """

    def __init__(self, n_atoms, func, *args, **kwargs):
        """
        Arguments
        ---------
        n_atoms   : int
                  total number of atoms
        func      : python method
                  inner function computing T*mu
        args      :
                  arguments provided to `func`
        kwargs    :
                  keyword arguments provided to `func`
        """
        # super().__init__(dtype, self.shape)
        # The `shape` and `dtype` must be implemented.
        self.shape = (n_atoms * 3, n_atoms * 3)
        # self.dtype = float #np.float64
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def _matvec(self, x):
        return self.func(x, *self.args, **self.kwargs).ravel()

    def matvec(self, x):
        x = np.asanyarray(x)
        m, n = self.shape
        if x.shape[0] != n:
            if x.shape[0] * x.shape[1] != n:
                raise ValueError("dimension mismatch")
        elif x.shape == (n, 1):
            x = x.reshape(int(n / 3), 3)
        elif x.shape == (n,):
            x = x.reshape(int(n / 3), 3)
        y = self._matvec(x)
        y = np.asarray(y)
        return y


def tmu_fortran_wrapper(
    mu, alpha, thole, pol_coords, iscreen=1, nn_list=None, dodiag=False
):
    """
    Wrapper to call FORTRAN code that computes T*mu
    Arguments
    ---------
    mu         : ndarray, (num_pol_atoms, 3)
               Induced dipoles on polarizable atoms
    alpha      : ndarray, (num_pol_atoms,)
               Polarizabilities of polarizable atoms
    thole      : float
               Thole factor
    pol_coords : ndarray, (num_pol_atoms, 3)
               Cartesian coordinates of polarizable atoms
    iscreen    : int
               Whether to include screening.
               0 means no screening
               1 means linear screening
               2 means exponential screening
               Note that each term should go with a dedicated
               thole factor. We only support iscreen=1 now.
    nn_list    : ndarray, (num_pol_atoms, max_neighbors)
               Neighbor list with second and third neighbors.
    dodiag     : bool
               Whether to include the diagonal contribution.
    """
    E = tmu_fortran(mu.T, alpha, thole, nn_list.T, pol_coords.T, iscreen).T
    if dodiag:
        E += mu / alpha[:, None]
    return E


def _compute_thole_factor(alpha):
    """
    Compute the thole factor associated with WangAL parameters.
    Arguments
    ---------
    alpha    : ndarray, (num_pol_atoms,)
             Atomic polarizabilities
    """
    fa = WANGAL_FACTOR**0.5
    return fa * alpha**1.0 / 6.0


def mmpol_coupling(
    coords,
    coords1,
    coords2,
    charges1,
    charges2,
    polarizabilities,
    env_mask,
    pol_threshold,
    full_connect,
):
    """
    Compute the MMPol contribution to the TrEsp Coulomb coupling.
    Arguments
    ---------
    coords           : ndarray, (num_atoms, 3)
                     Cartesian coordinates of every atom
    coords1          : ndarray, (num_atoms_1, 3)
                     Cartesian coordinates of the first molecule
    coords2          : ndarray, (num_atoms_2, 3)
                     Cartesian coordinates of the second molecule
    charges1         : ndarray, (num_atoms_1,)
                     Charges of the first molecule
    charges2         : ndarray, (num_atoms_2,)
                     Charges of the second molecule
    polarizabilities : ndarray, (num_atoms,)
                     Atomic polarizabilities
    env_mask         : ndarray, (num_atoms,)
                     Environment mask, True if the atom belongs to the environment,
                     False otherwise.
    pol_threshold    : float
                     Polarization threshold
    full_connect     : ndarray, (num_atoms, max_connections)
                     Connectivity matrix
    """
    # Get the `cut_mask`, or polarization mask, that selects the polarizable
    # atoms within the given cutoff
    cut_mask = _polarization_cutoff(
        coords, env_mask, pol_threshold, full_connect, polarizabilities
    )
    # Compute the neighbor list of polarizable atoms, needed for the fortran routine
    nn_list = _build_pol_neighbor_list(full_connect, cut_mask)
    pol_coords = coords[cut_mask]
    alpha = polarizabilities[cut_mask]
    # Convert to Bohr
    pol_coords, coords1, coords2 = _angstrom2bohr(pol_coords, coords1, coords2)
    # Compute the electric fields of the two
    # molecules at the positions of polarizable atoms
    electric_field1 = electric_field(pol_coords, coords1, charges1)
    electric_field2 = electric_field(pol_coords, coords2, charges2)
    alpha3 = np.column_stack((alpha, alpha, alpha))
    num_atoms = len(alpha)
    # We only support WangAL polar model for now
    thole = _compute_thole_factor(alpha)
    # Define linear operators for TMu and preconditioner
    TT = TmuOperator(
        n_atoms=num_atoms,
        func=tmu_fortran_wrapper,
        alpha=alpha,
        thole=thole,
        pol_coords=pol_coords,
        iscreen=1,
        nn_list=nn_list,
        dodiag=True,
    )
    preconditioner = LinearOperator(TT.shape, matvec=lambda x: alpha3.ravel() * x)
    # Solve preconditioned Conjugate Gradient
    mu, info = cg(TT, electric_field1.ravel(), M=preconditioner, tol=1e-4)
    Vmmp = -np.sum(mu * electric_field2.ravel())
    return Vmmp * HARTREE2CM_1


# Alias for site energies
# The function is the same, is only called with coords1=coords2
# and charges1=charges2
mmpol_site_contribution = mmpol_coupling


# =============================================================================
# High Level Wrappers
# =============================================================================


def compute_mmpol_coupling(
    trajectory, coords, charges, residue_ids, masks, pair, pol_threshold
):
    """
    Compute MMPol contribution to the Coulomb coupling of a pair of chromophores.
    Arguments
    ---------
    trajectory    : pytraj.Trajectory
                    Trajectory object.
    coords        : list of ndarray (num_frames, num_atoms, 3)
                    Coordinates of the chromophores pair
    charges       : list of ndarray, (num_atoms,)
                    Charges of the chromophores pair
    residue_ids   : list of int
                    Residue IDs
    masks         : list of str
                    Mask of the chromophore
    pol_threshold : float
                    Polarization threshold
    Returns
    -------
    mmpol_couplings : ndarray, (num_frames,)
                      MMPol contributions to the Coulomb Coupling.
    """
    topology = trajectory.topology
    num_frames = trajectory.n_frames
    connectivity = build_connectivity_matrix(topology, count_as="fortran")
    polarizabilities = compute_polarizabilities(topology)
    mmpol_couplings = []
    resid1 = residue_ids[pair[0]]
    resid2 = residue_ids[pair[1]]
    iterator = pbar(
        pt.iterframe(trajectory),
        total=num_frames,
        desc=f": {resid1}_{resid2} MMPol contribution:",
        ncols=79,
    )
    # iterator = pt.iterframe(trajectory)
    for i, frame in enumerate(iterator):
        full_coords = frame.xyz.copy()
        coords1 = coords[pair[0]][i]
        coords2 = coords[pair[1]][i]
        charges1 = charges[pair[0]][i]
        charges2 = charges[pair[1]][i]
        # Compute the environment mask
        env_mask = compute_environment_mask(
            topology,
            masks[pair[0]],
            masks[pair[1]],
        )
        V_mmp = mmpol_coupling(
            full_coords,
            coords1,
            coords2,
            charges1,
            charges2,
            polarizabilities,
            env_mask,
            pol_threshold,
            connectivity,
        )
        mmpol_couplings.append(V_mmp)
    mmpol_couplings = np.asarray(mmpol_couplings)
    return mmpol_couplings


def compute_mmpol_couplings(
    trajectory, coords, charges, residue_ids, masks, pairs, pol_threshold
):
    """
    Compute MMPol contribution to the Coulomb coupling along a trajectory.
    Arguments
    ---------
    trajectory    : pytraj.Trajectory
                    Trajectory object.
    coords        : list of ndarray (num_frames, num_atoms, 3)
                    Coordinates of the chromophores pairs
    charges       : list of ndarray, (num_atoms,)
                    Charges of the chromophores pairs
    residue_ids   : list of int
                    Residue IDs
    masks         : list of str
                    Mask of the chromophores
    pol_threshold : float
                    Polarization threshold
    Returns
    -------
    mmpol_couplings : list of ndarray, (num_frames,)
                    List of MMPol contributions to the Coulomb Coupling.
    """
    mmpol_couplings = []
    for pair in pairs:
        mmp_coup = compute_mmpol_coupling(
            trajectory,
            coords=coords,
            charges=charges,
            residue_ids=residue_ids,
            masks=masks,
            pair=pair,
            pol_threshold=pol_threshold,
        )
        mmpol_couplings.append(mmp_coup)
    return mmpol_couplings


def compute_mmpol_site_energy(
    trajectory, coords, charges, residue_id, mask, pol_threshold
):
    """
    Compute MMPol contribution to the site energy of a single
    chromophore.
    Arguments
    ---------
    trajectory    : pytraj.Trajectory
                    Trajectory object.
    coords        : ndarray (num_frames, num_atoms, 3)
                    Coordinates of the chromophore
    charges       : ndarray, (num_atoms,)
                    Charges of the chromophore
    residue_id    : int
                    Residue ID
    mask          : str
                    Mask of the chromophore
    pol_threshold : float
                    Polarization threshold
    Returns
    -------
    mmpol_site_contribs : ndarray, (num_frames,)
                          MMPol shifts to the site energy.
    """
    topology = trajectory.topology
    num_frames = trajectory.n_frames
    connectivity = build_connectivity_matrix(topology, count_as="fortran")
    polarizabilities = compute_polarizabilities(topology)
    mmpol_site_contribs = []
    iterator = pbar(
        pt.iterframe(trajectory),
        total=num_frames,
        desc=f": {residue_id} MMPol contribution:",
        ncols=79,
    )
    # iterator = pt.iterframe(trajectory)
    for i, frame in enumerate(iterator):
        full_coords = frame.xyz.copy()
        coords1 = coords[i]
        coords2 = coords[i]
        charges1 = charges[i]
        charges2 = charges[i]
        # Compute the environment mask
        env_mask = compute_environment_mask(
            topology,
            mask,
            mask,
        )
        site_mmp = mmpol_site_contribution(
            full_coords,
            coords1,
            coords2,
            charges1,
            charges2,
            polarizabilities,
            env_mask,
            pol_threshold,
            connectivity,
        )
        mmpol_site_contribs.append(site_mmp)
    mmpol_site_contribs = np.asarray(mmpol_site_contribs)
    return mmpol_site_contribs


def compute_mmpol_site_energies(
    trajectory, coords, charges, residue_ids, masks, pol_threshold
):
    """
    Compute the MMPol contribution to the site energy of a series of
    chromophores.
    Arguments
    ---------
    trajectory    : pytraj.Trajectory
                    Trajectory object.
    coords        : list of ndarray (num_frames, num_atoms, 3)
                    Coordinates of the chromophores
    charges       : list of ndarray (num_atoms,)
                    Charges of the chromophores
    residue_ids   : list of int
                    Residue IDs of the chromophores
    masks         : list of str
                    Masks of the chromophores
    pol_threshold : float
                    Polarization threshold
    Returns
    -------
    mmpol_site_contribs : list of ndarray (num_frames,)
                          MMPol shifts to the site energy.
    """
    mmpol_site_energies = []
    for coord, charge, residue_id, mask in zip(coords, charges, residue_ids, masks):
        mmp_site = compute_mmpol_site_energy(
            trajectory,
            coords=coord,
            charges=charge,
            residue_id=residue_id,
            mask=mask,
            pol_threshold=pol_threshold,
        )
        mmpol_site_energies.append(mmp_site)
    return mmpol_site_energies
