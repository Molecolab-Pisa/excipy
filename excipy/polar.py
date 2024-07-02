from __future__ import annotations
from typing import Union, List, Tuple, Iterator, Any

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
from .elec import (
    read_electrostatics,
    link_atom_smear,
    WANGAL_FACTOR,
    electric_field,
)
from .selection import spherical_cutoff, _whole_residues_mm_cutoff

Trajectory = Union[pt.Trajectory, pt.TrajectoryIterator]


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


def _read_polar_and_smear(
    topology: pt.Topology,
    qm_idx: np.ndarray,
    db: str = None,
    mol2: str = None,
    smear_link: bool = True,
    turnoff_mask: str = None,
) -> np.ndarray:
    """reads the polarizabilities and applies a smearing for link atoms

    Reads the polarizabilities from the top or the database, and applies
    a smearing if the QM regions cuts through a bond. The mol2 is used
    to recognize e.g. residues at the ends of the protein chain.
    """
    # read only the polarizabilities now
    _, _, polarizabilities = read_electrostatics(
        topology,
        db=db,
        mol2=mol2,
        warn=True,
    )
    if turnoff_mask:
        # shut down polarizability for turnoff atoms
        toff_idx = topology.select(turnoff_mask)
        polarizabilities[toff_idx] = 0.0
    # kept here for back compatibility, but the
    # correct thing to do is to apply the smearing
    if smear_link:
        _, _, polarizabilities = link_atom_smear(
            topology,
            qm_idx,
            np.zeros_like(polarizabilities),
            np.zeros_like(polarizabilities),
            polarizabilities,
        )
    return polarizabilities


# ============================================================================
# Handy functions to compute the cutoff
# ============================================================================


def _pol_from_spherical_cutoff(
    topology: pt.Topology,
    qm_mask: str,
    qm_coords: np.ndarray,
    full_coords: np.ndarray,
    pol_cut: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """spherical cutoff

    Applies a spherical cut around the QM region.
    All the atoms within the pol_cut distance are retained.
    This means that many MM residues will likely be split in a half.
    """
    env_mask = compute_environment_mask(topology, qm_mask, qm_mask)
    env_idx = np.arange(topology.n_atoms)[env_mask]
    _, pol_coords, pol_idx, _ = spherical_cutoff(
        source_coords=qm_coords,
        ext_coords=full_coords[env_idx],
        cutoff=pol_cut,
    )
    # indices of the pol atoms in the full system
    pol_idx = env_idx[pol_idx]
    return pol_coords, pol_idx


def _pol_from_whole_cutoff(
    topology: pt.Topology, qm_mask: str, full_coords: np.ndarray, pol_cut: float
) -> Tuple[np.ndarray, np.ndarray]:
    """whole residues cuto

    Applies a cut around the QM region where MM residues are
    kept intact. More expensive than performing a spherical cutoff.
    """
    _, pol_coords, _, pol_idx, _ = _whole_residues_mm_cutoff(
        topology=topology,
        coords=full_coords,
        qm_mask=qm_mask,
        mm_cut=pol_cut,
    )
    return pol_coords, pol_idx


def _apply_pol_cutoff(
    topology: pt.Topology,
    qm_mask: str,
    qm_coords: np.ndarray,
    full_coords: np.ndarray,
    pol_cut: float,
    strategy: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """polarization cut

    Applies a cut to the polarization part according to the
    selected strategy. If the strategy is spherical, a spherical
    cut is performed. If the strategy is whole, residues of the
    environment will be kept intact.
    """
    if strategy.lower() == "spherical":
        pol_coords, pol_idx = _pol_from_spherical_cutoff(
            topology,
            qm_mask,
            qm_coords,
            full_coords,
            pol_cut,
        )
    elif strategy == "whole":
        pol_coords, pol_idx = _pol_from_whole_cutoff(
            topology,
            qm_mask,
            full_coords,
            pol_cut,
        )
    else:
        raise ValueError('cutoff strategy must be either "spherical" or "whole"')
    return pol_coords, pol_idx


def _exclude_zero_polar(
    polarizabilities: np.ndarray, pol_idx: np.ndarray, pol_coords: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """exclude pol atoms with zero polarizability

    Exclude pol atoms with zero polarizability to avoid problems
    when setting up the linear system.
    """
    # polarizabilities for the pol atoms only
    alpha = polarizabilities[pol_idx]
    # exclude pol atoms with zero polarizability
    nonzero = alpha != 0.0
    pol_idx = pol_idx[nonzero]
    pol_coords = pol_coords[nonzero]
    alpha = alpha[nonzero]
    return pol_coords, pol_idx, alpha


def _pol_cut_mask(num_atoms: int, pol_idx: np.ndarray) -> np.ndarray:
    cut_mask = np.zeros(num_atoms, dtype=bool)
    cut_mask[pol_idx] = True
    return cut_mask


def _solve_mu_ind(
    alpha: np.ndarray,
    electric_field: np.ndarray,
    pol_coords: np.ndarray,
    nn_list: np.ndarray,
    iscreen: int = 1,
    dodiag: bool = True,
) -> Tuple[np.ndarray, Any]:
    """solves for the induced dipoles

    Solves for the induced dipoles using a preconditioned conjugate gradient
    solver.
    """
    n_pol_atoms = len(alpha)
    thole = _compute_thole_factor(alpha)
    alpha3 = np.column_stack((alpha, alpha, alpha))
    TT = TmuOperator(
        n_atoms=n_pol_atoms,
        func=tmu_fortran_wrapper,
        alpha=alpha,
        thole=thole,
        pol_coords=pol_coords,
        iscreen=iscreen,
        nn_list=nn_list,
        dodiag=dodiag,
    )
    precond = LinearOperator(TT.shape, matvec=lambda x: alpha3.ravel() * x)
    mu, info = cg(TT, electric_field.ravel(), M=precond, tol=1e-4)
    return mu, info


def _set_iterator(trajectory: Trajectory, desc: str) -> Iterator:
    return pbar(
        pt.iterframe(trajectory),
        total=trajectory.n_frames,
        desc=desc,
        ncols=79,
    )


# ============================================================================
# Coupling
# ============================================================================


def mmpol_coup_lr(
    trajectory: Trajectory,
    coords1: np.ndarray,
    coords2: np.ndarray,
    charges1: np.ndarray,
    charges2: np.ndarray,
    residue_id1: str,
    residue_id2: str,
    mask1: str,
    mask2: str,
    pol_threshold: float,
    db: str = None,
    mol2: str = None,
    cut_strategy: str = "spherical",
    smear_link: bool = True,
    turnoff_mask: str = None,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """V_LR along a trajectory

    Computes the linear response term of the electronic coupling along
    a trajectory for a specific pair of chromophores.
    """
    topology = trajectory.topology
    num_atoms = topology.n_atoms
    qm1_idx = topology.select(mask1)
    qm2_idx = topology.select(mask2)
    qm_idx = np.concatenate((qm1_idx, qm2_idx))
    # read polarizabilities + apply link atom smearing
    polarizabilities = _read_polar_and_smear(
        topology, qm_idx, db, mol2, smear_link, turnoff_mask
    )
    connectivity = build_connectivity_matrix(topology, count_as="fortran")
    iterator = _set_iterator(
        trajectory, f": {residue_id1}_{residue_id2} MMPol Linear Response:"
    )
    coups = []
    dipoles = []
    for i, frame in enumerate(iterator):
        full_coords = frame.xyz.copy()
        qm1_coords = coords1[i]
        qm2_coords = coords2[i]
        qm1_charges = charges1[i]
        qm2_charges = charges2[i]
        pair_mask = make_pair_mask(mask1, mask2)
        qm_coords = np.concatenate((qm1_coords, qm2_coords), axis=0)
        pol_coords, pol_idx = _apply_pol_cutoff(
            topology, pair_mask, qm_coords, full_coords, pol_threshold, cut_strategy
        )
        pol_coords, pol_idx, alpha = _exclude_zero_polar(
            polarizabilities, pol_idx, pol_coords
        )
        cut_mask = _pol_cut_mask(num_atoms, pol_idx)
        nn_list = _build_pol_neighbor_list(connectivity, cut_mask)
        pol_coords, qm1_coords, qm2_coords = _angstrom2bohr(
            pol_coords, qm1_coords, qm2_coords
        )
        elec_field1 = electric_field(pol_coords, qm1_coords, qm1_charges)
        elec_field2 = electric_field(pol_coords, qm2_coords, qm2_charges)
        mu, info = _solve_mu_ind(
            alpha, elec_field1, pol_coords, nn_list, iscreen=1, dodiag=True
        )
        Vmmp = -np.sum(mu * elec_field2.ravel())
        Vmmp *= HARTREE2CM_1

        coups.append(Vmmp)
        dipoles.append(mu)

    return np.array(coups), dipoles


def batch_mmpol_coup_lr(
    trajectory: Trajectory,
    coords: List[np.ndarray],
    charges: List[np.ndarray],
    residue_ids: List[str],
    masks: List[str],
    pairs: List[List[int]],
    pol_threshold: float,
    cut_strategy: str = "spherical",
    smear_link: bool = True,
    db: str = None,
    mol2: str = None,
    turnoff_mask: str = None,
) -> Tuple[List[np.ndarray], List[List[np.ndarray]]]:
    """V_LR along a trajectory

    Computes the Linear Response term of the electronic coupling
    along a trajectory and for all the pairs specified in `pairs`.
    """
    out_lr = []
    out_mu = []
    for i1, i2 in pairs:
        coup_lr, mu = mmpol_coup_lr(
            trajectory,
            coords1=coords[i1],
            coords2=coords[i2],
            charges1=charges[i1],
            charges2=charges[i2],
            residue_id1=residue_ids[i1],
            residue_id2=residue_ids[i2],
            mask1=masks[i1],
            mask2=masks[i2],
            pol_threshold=pol_threshold,
            db=db,
            mol2=mol2,
            cut_strategy=cut_strategy,
            smear_link=smear_link,
            turnoff_mask=turnoff_mask,
        )
        out_lr.append(coup_lr)
        out_mu.append(mu)
    return out_lr, out_mu


# ============================================================================
# Site energy
# ============================================================================


def mmpol_site_lr(
    trajectory: Trajectory,
    coords: np.ndarray,
    charges: np.ndarray,
    residue_id: str,
    mask: str,
    pol_threshold: float,
    db: str = None,
    mol2: str = None,
    cut_strategy: str = "spherical",
    smear_link: bool = True,
    turnoff_mask: str = None,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """E_LR along a trajectory

    Computes the Linear Response term of the excitation energy along
    a trajectory for a single residue.
    """
    topology = trajectory.topology
    num_atoms = topology.n_atoms
    qm_idx = topology.select(mask)
    # read polarizabilities + apply link atom smearing
    polarizabilities = _read_polar_and_smear(
        topology, qm_idx, db, mol2, smear_link, turnoff_mask
    )
    # connectivity is needed only for the tmu solver
    connectivity = build_connectivity_matrix(topology, count_as="fortran")
    iterator = _set_iterator(trajectory, f": {residue_id} MMPol Linear Response:")

    energies = []
    dipoles = []
    for i, frame in enumerate(iterator):
        full_coords = frame.xyz.copy()
        qm_coords = coords[i]
        qm_charges = charges[i]
        pol_coords, pol_idx = _apply_pol_cutoff(
            topology, mask, qm_coords, full_coords, pol_threshold, cut_strategy
        )
        pol_coords, pol_idx, alpha = _exclude_zero_polar(
            polarizabilities, pol_idx, pol_coords
        )
        cut_mask = _pol_cut_mask(num_atoms, pol_idx)
        nn_list = _build_pol_neighbor_list(connectivity, cut_mask)
        pol_coords, qm_coords = _angstrom2bohr(pol_coords, qm_coords)
        elec_field = electric_field(pol_coords, qm_coords, qm_charges)
        # mu induced by electric field
        mu, info = _solve_mu_ind(
            alpha, elec_field, pol_coords, nn_list, iscreen=1, dodiag=True
        )
        # interaction with source field
        Vmmp = -np.sum(mu * elec_field.ravel())
        Vmmp *= HARTREE2CM_1

        # collect
        energies.append(Vmmp)
        dipoles.append(mu)

    return np.array(energies), dipoles


def batch_mmpol_site_lr(
    trajectory: Trajectory,
    coords: List[np.ndarray],
    charges: List[np.ndarray],
    residue_ids: List[str],
    masks: List[str],
    pol_threshold: float,
    cut_strategy: str = "spherical",
    smear_link: bool = True,
    db: str = None,
    mol2: str = None,
    turnoff_mask: str = None,
) -> Tuple[List[np.ndarray], List[List[np.ndarray]]]:
    """E_LR along a trajectory

    Computes the Linear Response term of the energy along a trajectory
    for all the residues listed in `residue_ids`.
    """
    out_lr = []
    out_mu = []
    for coord, charge, residue_id, mask in zip(coords, charges, residue_ids, masks):
        site_lr, mu = mmpol_site_lr(
            trajectory,
            coords=coord,
            charges=charge,
            residue_id=residue_id,
            mask=mask,
            pol_threshold=pol_threshold,
            db=db,
            mol2=mol2,
            cut_strategy=cut_strategy,
            smear_link=smear_link,
            turnoff_mask=turnoff_mask,
        )
        out_lr.append(site_lr)
        out_mu.append(mu)
    return out_lr, out_mu
