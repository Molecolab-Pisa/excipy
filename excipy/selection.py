from __future__ import annotations
from typing import Tuple

import pytraj as pt
import numpy as np
from scipy.spatial.distance import cdist

from .clib.retain_full_residues import retain_full_residues_cy


PytrajTopology = pt.Topology


# ============================================================
# Functions to select residues around a collection of "source" atoms
# ============================================================


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
    ext_topology: PytrajTopology,
    source_coords: np.ndarray,
    ext_coords: np.ndarray,
    cutoff: float,
):
    """applies the cutoff without cutting the residues

    Applies a given cutoff around the source atoms so that each
    external residue is taken as a whole.

    Args:
        ext_topology: pytraj topology of just the external part
        source_coords: coordinates of the source part
        ext_coords: coordinates of the external part
        cutoff: cutoff for the external part
    """
    ext_idx = np.arange(ext_topology.n_atoms)
    residues_array = _get_residues_array(ext_topology, ext_idx)
    # Compute the distances between each atom of the pigment
    # pair and every other atom of the environment
    # dist is of shape (num_env_atoms, num_pair_atoms)
    dist = cdist(ext_coords, source_coords)
    # For each environment atom, find the minimum distance
    # from the pair coordinates
    # min_dist is of shape (num_env_atoms,)
    min_dist = np.min(dist, axis=1)
    # Keep atoms within the cutoff
    ext_mask = min_dist <= cutoff
    cut_mask = np.zeros(len(min_dist), dtype=bool)
    cut_mask[ext_mask] = True
    # mm mask now has the residues as a whole, within the cutoff
    ext_mask = retain_full_residues_cy(
        cut_mask.astype(np.intc), residues_array.astype(np.intc)
    ).astype(bool)
    num_ext = sum(ext_mask)
    ext_coords = ext_coords[ext_mask]
    # original indices of the selected mm atoms in the full mm topology
    ext_idx = ext_idx[ext_mask]
    # amber mask counts from 1
    ext_mask = "@" + ",".join((ext_idx + 1).astype(str))
    ext_top = ext_topology[ext_mask]
    return num_ext, ext_coords, ext_top, ext_idx


def _spherical_cutoff(
    source_coords: np.ndarray, ext_coords: np.ndarray, cutoff: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """applies a spherical cutoff around each source atom

    Applies a spherical cutoff around each source atoms.
    Residues of the external part may be truncated.
    Note: the sphere is around each atom, i.e., the global
          cut is not spherical.

    Args:
        source_coords: coordinates of the source part
        ext_coords: coordinates of the external part
        cutoff: cutoff for the external part
    """
    ext_idx = np.arange(ext_coords.shape[0])
    dist = cdist(ext_coords, source_coords)
    min_dist = np.min(dist, axis=1)
    ext_mask = min_dist <= cutoff
    ext_idx = ext_idx[ext_mask]
    ext_coords = ext_coords[ext_mask]
    num_ext = ext_idx.shape[0]
    return num_ext, ext_coords, ext_idx


# ============================================================
# Higher level: specialized to MM and Pol parts
# ============================================================


def _whole_residues_mm_cutoff(
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
        ext_topology=mm_top,
        source_coords=qm_coords,
        ext_coords=mm_coords,
        cutoff=mm_cut,
    )

    # indices in the original topology
    mm_idx = mm_idx[mm_sel]

    return qm_coords, mm_coords, qm_idx, mm_idx, mm_top


def _whole_residues_pol_cutoff(
    mm_topology: PytrajTopology,
    qm_coords: np.ndarray,
    mm_coords: np.ndarray,
    pol_cut: float,
):
    """applies the pol cutoff

    Applies the Pol cutoff around the QM mask, keeping the Pol residues as a whole
    (i.e., without cutting the Pol residues)

    Note: this must be done after a call to whole_residues_mm_cutoff
          to save time!

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
        ext_topology=mm_topology,
        source_coords=qm_coords,
        ext_coords=mm_coords,
        cutoff=pol_cut,
    )
    return pol_coords, pol_top, pol_idx
