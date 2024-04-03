from __future__ import annotations

import pytraj as pt
import numpy as np
from scipy.spatial.distance import cdist

from .clib.retain_full_residues import retain_full_residues_cy


PytrajTopology = pt.Topology


def _get_residues_array(topology: PytrajTopology, mm_indices: np.ndarray) -> np.ndarray:
    """compute the residues_array

    Computes an array storing the indices of the first atom of each residue.
    The last element is the number of MM atoms.
    It is assumed that residues appear sequentially, i.e., there is no atom of
    residue j in the middle of atoms of residue i.

    Args:
        topology: pytraj topology
        mm_indices: indices of the MM atoms
    """
    num_mm = np.array([len(mm_indices)])
    _, indices = np.unique(
        np.array([topology.atom(i).resid for i in mm_indices]), return_index=True
    )
    # This array is passed to C code: ensure it is an array of C integers
    residues_array = np.concatenate([indices, num_mm])  # .astype(np.intc)
    return residues_array


def _cutoff_with_whole_residues(
    mm_topology: PytrajTopology,
    qm_coords: np.ndarray,
    mm_coords: np.ndarray,
    cutoff: float,
):
    """applies the cutoff without cutting the residues

    Applies a given cutoff around the QM atoms so that each MM residue is
    taken as a whole.

    Args:
        mm_topology: pytraj topology of just the mm part
        qm_coords: coordinates of the qm part
        mm_coords: coordinates of the mm part
        cutoff: cutoff for the mm part
    """
    mm_idx = np.arange(mm_topology.n_atoms)
    residues_array = _get_residues_array(mm_topology, mm_idx)
    # Compute the distances between each atom of the pigment
    # pair and every other atom of the environment
    # dist is of shape (num_env_atoms, num_pair_atoms)
    dist = cdist(mm_coords, qm_coords)
    # For each environment atom, find the minimum distance
    # from the pair coordinates
    # min_dist is of shape (num_env_atoms,)
    min_dist = np.min(dist, axis=1)
    # Keep atoms within the cutoff
    mm_mask = min_dist <= cutoff
    cut_mask = np.zeros(len(min_dist), dtype=bool)
    cut_mask[mm_mask] = True
    # mm mask now has the residues as a whole, within the cutoff
    mm_mask = retain_full_residues_cy(
        cut_mask.astype(np.intc), residues_array.astype(np.intc)
    ).astype(bool)
    num_mm = sum(mm_mask)
    mm_coords = mm_coords[mm_mask]
    # original indices of the selected mm atoms in the full mm topology
    mm_idx = mm_idx[mm_mask]
    # amber mask counts from 1
    mm_mask = "@" + ",".join((mm_idx + 1).astype(str))
    mm_top = mm_topology[mm_mask]
    return num_mm, mm_coords, mm_top, mm_idx


def _mm_cutoff(
    topology: PytrajTopology, coords: np.ndarray, qm_mask: str, mm_cut: float
):
    """applies the mm cutoff

    Applies the MM cutoff around the QM mask, keeping the MM residues as a whole
    (i.e., without cutting the MM residues).

    Args:
        topology: pytraj topology of the whole system
        coords: coordinates of the whole system, (n_atoms, 3)
        qm_mask: amber mask to select the QM part
        mm_cut: cutoff
    Returns:
        qm_coords: coordinates of the QM part, (num_qm, 3)
        mm_coords: coordinates of the MM part, (num_mm, 3)
        qm_idx: indices of the QM atoms in the original topology, (num_qm,)
        mm_idx: indices of the MM atoms in the original topology, (num_mm,)
        mm_top: pytraj topology containing only the MM part
    """
    mm_mask = "!" + qm_mask
    mm_top = topology[mm_mask]
    qm_idx = topology.select(qm_mask)
    mm_idx = topology.select(mm_mask)
    qm_coords = coords[qm_idx]
    mm_coords = coords[mm_idx]

    num_mm, mm_coords, mm_top, mm_sel = _cutoff_with_whole_residues(
        mm_topology=mm_top,
        qm_coords=qm_coords,
        mm_coords=mm_coords,
        cutoff=mm_cut,
    )

    # indices in the original topology
    mm_idx = mm_idx[mm_sel]

    return qm_coords, mm_coords, qm_idx, mm_idx, mm_top


def _pol_cutoff(
    mm_topology: PytrajTopology,
    qm_coords: np.ndarray,
    mm_coords: np.ndarray,
    pol_cut: float,
):
    """applies the pol cutoff

    Applies the Pol cutoff around the QM mask, keeping the Pol residues as a whole
    (i.e., without cutting the Pol residues)

    Note: this must be done after a call to _mm_cutoff to save time!

    Args:
        mm_topology: pytraj topology containing only the MM atoms
        qm_coords: coordinates of the QM part, (num_qm, 3)
        mm_coords: coordinates of the MM part, (num_mm, 3)
        pol_cut: cutoff
    Returns:
        pol_coords: coordinates of the Pol part, (num_pol, 3)
        pol_top: pytraj topology containing only the Pol part
        pol_idx: indices of the Pol atoms in the MM topology, (num_pol,)
    """
    # note: pol_idx here tells you the indeces of the pol atoms
    # *in the mm topology*, not the indeces of the pol atoms in
    # the full topology (i.e., if the first atom of the mm topology
    # is the 10th atom, and is polarizable, then it has pol_idx 0)
    num_pol, pol_coords, pol_top, pol_idx = _cutoff_with_whole_residues(
        mm_topology=mm_topology,
        qm_coords=qm_coords,
        mm_coords=mm_coords,
        cutoff=pol_cut,
    )
    return pol_coords, pol_top, pol_idx
