from __future__ import annotations

import numpy as np
from scipy.sparse.linalg import LinearOperator, cg
import pytraj as pt
from .tmu import tmu as tmu_fortran
from .util import ANG2BOHR, HARTREE2CM_1
from .util import (
    make_pair_mask,
    build_connectivity_matrix,
    _count_as,
    _pad_list_of_lists,
    pbar,
)
from .elec import compute_polarizabilities, WANGAL_FACTOR, electric_field
from .selection import spherical_cutoff


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


def _angstrom2bohr(*arrays):
    """
    Convert from angstrom to bohr units an arbitrary number of arrays.
    """
    return (a * ANG2BOHR for a in arrays)


# =============================================================================
# MMPol Manipulation Functions
# =============================================================================


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


def _compute_cut_mask(env_mask: np.ndarray, pol_idx: np.ndarray):
    cut_mask = np.zeros(len(env_mask), dtype=bool)
    selected = np.where(env_mask)[0][pol_idx]
    cut_mask[selected] = True
    return cut_mask


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
    num_pol, pol_coords, pol_idx = spherical_cutoff(
        source_coords=np.concatenate((coords1, coords2), axis=0),
        ext_coords=coords[env_mask],
        cutoff=pol_threshold,
    )
    cut_mask = _compute_cut_mask(env_mask=env_mask, pol_idx=pol_idx)
    # exclude atoms with polarizability of zero
    alpha = polarizabilities[cut_mask]
    cut_mask[polarizabilities == 0] = False
    pol_coords = pol_coords[alpha != 0]
    alpha = alpha[np.where(alpha != 0)[0]]
    # Compute the neighbor list of polarizable atoms, needed for the fortran routine
    nn_list = _build_pol_neighbor_list(full_connect, cut_mask)
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


def mmpol_site_contribution(
    coords,
    coords1,
    charges1,
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
    num_pol, pol_coords, pol_idx = spherical_cutoff(
        source_coords=coords1,
        ext_coords=coords[env_mask],
        cutoff=pol_threshold,
    )
    cut_mask = _compute_cut_mask(env_mask=env_mask, pol_idx=pol_idx)
    # exclude atoms with polarizability of zero
    alpha = polarizabilities[cut_mask]
    cut_mask[polarizabilities == 0] = False
    pol_coords = pol_coords[alpha != 0]
    alpha = alpha[np.where(alpha != 0)[0]]
    # Compute the neighbor list of polarizable atoms, needed for the fortran routine
    nn_list = _build_pol_neighbor_list(full_connect, cut_mask)
    # Convert to Bohr
    pol_coords, coords1 = _angstrom2bohr(pol_coords, coords1)
    # Compute the electric fields of the two
    # molecules at the positions of polarizable atoms
    electric_field1 = electric_field(pol_coords, coords1, charges1)
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
    Vmmp = -np.sum(mu * electric_field1.ravel())
    return Vmmp * HARTREE2CM_1


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
        charges1 = charges[i]
        # Compute the environment mask
        env_mask = compute_environment_mask(
            topology,
            mask,
            mask,
        )
        site_mmp = mmpol_site_contribution(
            full_coords,
            coords1,
            charges1,
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
