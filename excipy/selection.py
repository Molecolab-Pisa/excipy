from __future__ import annotations
from typing import Tuple, List, Optional

import pytraj as pt
import numpy as np
from scipy.spatial.distance import cdist

from .clib.retain_full_residues import retain_full_residues_cy
from .util import pbar


PytrajTopology = pt.Topology


# ============================================================
# Functions to select sets of collections of atoms
# ============================================================


def squared_distances(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """squared euclidean distances

    This is a memory-efficient (and fast) implementation of the
    calculation of squared euclidean distances. Euclidean distances
    between `x1` of shape (n_samples1, n_feats) and `x2` of shape
    (n_samples2, n_feats) is evaluated by using the "euclidean distances
    trick":

        dist = X1 @ X1.T - 2 X1 @ X2.T + X2 @ X2.T

    Note: this function evaluates distances between batches of points

    Args:
        x1: first set of points, (n_samples1, n_feats)
        x2: second set of points, (n_samples2, n_feats)
    Returns:
        dist: squared distances between x1 and x2
    """
    jitter = 1e-12
    x1s = np.sum(np.square(x1), axis=-1)
    x2s = np.sum(np.square(x2), axis=-1)
    dist = x1s[:, np.newaxis] - 2 * np.dot(x1, x2.T) + x2s + jitter
    return dist


def euclidean_distances(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """euclidean distances

    Memory efficient and fast implementation of the calculation
    of squared euclidean distances.
    """
    return np.maximum(squared_distances(x1=x1, x2=x2), 1e-300) ** 0.5


def distance_geometric_centers(coords1: np.ndarray, coords2: np.ndarray) -> np.ndarray:
    """distance between geometric centers

    Computes the distance between the geometric centers of two sets of
    coordinates.

    Args:
        coords1: coordinates of the first set, (num_frames, num_atoms1, 3)
        coords2: coordinates of the second set, (num_frames, num_atoms2, 3)
    Returns:
        dist: distance between geometric centers, (num_frames,)
    """
    centers1 = np.mean(coords1, axis=1)
    centers2 = np.mean(coords2, axis=1)
    dist = np.sum((centers1 - centers2) ** 2, axis=-1) ** 0.5
    return dist


def compute_retain_list(
    coords: List[np.ndarray],
    residue_ids: List[str],
    pairs: List[List[int]],
    cutoff: float,
    verbose: Optional[bool] = False,
):
    """
    Compute a list of pairs that have to be retained (True)
    of excluded (False) on the basis of a distance cutoff
    between the geometric centers of two molecules

    Args:
        coords: list of atomic corodinates, each (num_frames, num_atoms, 3)
        residue_ids: lsit of names of the residues (used for printing only)
        pairs: list of indices of residues composing each pair, e.g.
               [[0, 1], [2, 3] specifies pairs 0-1 and 2-3.
        cutoff: pairs within the cutoff are included, others are excluded.
    Returns:
        retain_list: contains True if the pair is within the cutoff,
                     False otherwise.
    """
    retain_list = []

    if verbose:
        iterator = pbar(
            pairs,
            desc=": Evaluating distances",
            ncols=79,
        )
    else:
        iterator = pairs

    for pair in iterator:
        if verbose:
            iterator.set_postfix(pair=f"{residue_ids[pair[0]]}.{residue_ids[pair[1]]}")
        coords1 = coords[pair[0]]
        coords2 = coords[pair[1]]
        dist = distance_geometric_centers(coords1, coords2)
        # Decide on the basis of the mean distance
        # if the pair has to be retained or not
        criterion = np.min(dist)
        if criterion < cutoff:
            retain = True
        else:
            retain = False
        retain_list.append(retain)
    return retain_list


def apply_distance_cutoff(
    coords: List[np.ndarray],
    residue_ids: List[str],
    pairs: List[List[int]],
    cutoff: float,
    verbose: Optional[bool] = True,
):
    """applies a cutoff to select pairs of molecule

    Applies a cutoff that selects which pairs of coordinates should
    be retained given a list of pairs. The cutoff is applied on
    the geometric centers of the two pairs.

    Args:
        coords: list of atomic corodinates, each (num_frames, num_atoms, 3)
        residue_ids: lsit of names of the residues (used for printing only)
        pairs: list of indices of residues composing each pair, e.g.
               [[0, 1], [2, 3] specifies pairs 0-1 and 2-3.
        cutoff: pairs within the cutoff are included, others are excluded.
    Returns:
        pairs: list with the indices of the retained pairs, e.g.
               [[0, 1]] if only residues 0-1 are within the cutoff.
    """
    retain_list = compute_retain_list(
        coords, residue_ids, pairs, cutoff, verbose=verbose
    )
    pairs = [p for p, retain in zip(pairs, retain_list) if retain is True]
    return pairs


# ============================================================
# Functions to select residues around a collection of "source" atoms
# ============================================================


def get_residues_array(topology: PytrajTopology, mm_indices: np.ndarray) -> np.ndarray:
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


def cut_box(coords1: np.ndarray, coords2: np.ndarray, cutoff: float) -> np.ndarray:
    """cuts a cubic box around the atoms in coords1

    Cuts a cubic box around the atoms in coords1.
    The box extends a cutoff distance below/above the minimum/maximum
    position of atoms in coords1 in the three cartesian directions.

    Args:
        coords1: shape (n_atoms_1, 3)
        coords2: shape (n_atoms_2, 3)
        cutoff: cutoff value to cut the box
    Returns:
        idx0: True/False if an atom in coords2 is within/outside
              the cutted box, shape (n_atoms_2,)
    """
    # Select atoms in a box with length >= 2 * self.cutoff
    m1x = np.where(coords2[:, 0] < np.min(coords1[:, 0] - cutoff), 0, 1)
    m2x = np.where(coords2[:, 0] > np.max(coords1[:, 0] + cutoff), 0, 1)
    m1y = np.where(coords2[:, 1] < np.min(coords1[:, 1] - cutoff), 0, 1)
    m2y = np.where(coords2[:, 1] > np.max(coords1[:, 1] + cutoff), 0, 1)
    m1z = np.where(coords2[:, 2] < np.min(coords1[:, 2] - cutoff), 0, 1)
    m2z = np.where(coords2[:, 2] > np.max(coords1[:, 2] + cutoff), 0, 1)
    # True if an atom is within the box, False otherwise. We consider residues as a whole
    idx0 = np.min(np.row_stack([m1x, m2x, m1y, m2y, m1z, m2z]), axis=0).astype(bool)
    # Here we do not run a same_residue_as, so this cuts also the residues
    # we must then recover them later
    return idx0


def whole_residues_cutoff(
    source_coords: np.ndarray,
    ext_coords: np.ndarray,
    residues_array: np.ndarray,
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
    ext_idx = np.arange(ext_coords.shape[0])
    cut_mask = cut_box(source_coords, ext_coords, cutoff)
    orig_mask = cut_mask.copy()
    # Compute the distances between each atom of the pigment
    # pair and every other atom of the environment
    # dist is of shape (num_env_atoms, num_pair_atoms)
    dist = cdist(ext_coords[cut_mask], source_coords)
    # For each environment atom, find the minimum distance
    # from the pair coordinates
    # min_dist is of shape (num_env_atoms,)
    idx_cut = np.max(np.where(dist <= cutoff, 1, 0), axis=1).astype(bool)
    cut_mask[cut_mask] = idx_cut
    # mm mask now has the residues as a whole, within the cutoff
    ext_mask = retain_full_residues_cy(
        cut_mask.astype(np.intc), residues_array.astype(np.intc)
    ).astype(bool)
    # reuse the already-computed distances and just fill with the
    # missing values
    dist = _distances_diffmask(
        source_coords=source_coords,
        ext_coords=ext_coords,
        dist1=dist,
        cut0_mask=orig_mask,
        cut1_mask=cut_mask,
        cut2_mask=ext_mask,
    )
    # cut coordinates
    ext_coords = ext_coords[ext_mask]
    num_ext = ext_coords.shape[0]
    # original indices of the selected mm atoms in the full mm topology
    ext_idx = ext_idx[ext_mask]
    return num_ext, ext_coords, ext_idx, dist


def _distances_diffmask(
    source_coords: np.ndarray,
    ext_coords: np.ndarray,
    dist1: np.ndarray,
    cut0_mask: np.ndarray,
    cut1_mask: np.ndarray,
    cut2_mask: np.ndarray,
) -> np.ndarray:
    """
    Computes the distances only with those external coordinates that are not
    included in the first cut, but are included in the second cut. The zero
    cut takes into account additional, preliminary cuts, that are used to
    compute dist1.

    dist1 has shape (sum(cut0_mask), n_source), where sum(cut0_mask) is the
    number of external coordinates in the first cut. If there is not preliminary
    cut, provide cut0 as an array of True with length dist1.shape[0]
    """
    # find the final number of external coordinates (number of T in cut2_mask)
    n_ext = sum(cut2_mask)
    n_source = source_coords.shape[0]
    # create a distance matrix to be filled with the first set of distances and the new ones
    dist = np.zeros((n_ext, n_source))
    # fill the distance matrix with the precomputed distances of the first cut
    dist[cut1_mask[cut2_mask]] = dist1[cut1_mask[cut0_mask]]
    # find the entries that are present in cut2_mask but not in cut1_mask
    diff_mask = np.where(
        cut2_mask.astype(int) + cut1_mask.astype(int) == 1, True, False
    )
    # new set of distances, renaming dist1 as it's not used anymore
    dist1 = cdist(ext_coords[diff_mask], source_coords)
    # fill the empty entries of dist
    dist[~cut1_mask[cut2_mask]] = dist1
    return dist


def cut_topology(top: pt.Topology, idx: np.ndarray):
    # amber mask counts from 1
    mask = "@" + ",".join((idx + 1).astype(str))
    return top[mask]


def spherical_cutoff(
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

    Returns:
        num_ext: number of selected external atoms
        ext_coords: coordinates of the selected external atoms
        ext_idx: indices of the selected external atoms
        dist: distances between the selected external atoms
              and the source atoms
    """
    ext_idx = np.arange(ext_coords.shape[0])
    dist = cdist(ext_coords, source_coords)
    min_dist = np.min(dist, axis=1)
    ext_mask = min_dist <= cutoff
    ext_idx = ext_idx[ext_mask]
    ext_coords = ext_coords[ext_mask]
    dist = dist[ext_mask]
    num_ext = ext_idx.shape[0]
    return num_ext, ext_coords, ext_idx, dist


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
    mm_mask = "!(" + qm_mask + ")"
    mm_top = topology[mm_mask]
    qm_idx = topology.select(qm_mask)
    mm_idx = topology.select(mm_mask)
    qm_coords = coords[qm_idx]
    mm_coords = coords[mm_idx]

    residues_array = get_residues_array(
        topology=mm_top, mm_indices=np.arange(mm_coords.shape[0])
    )
    num_mm, mm_coords, mm_sel, _ = whole_residues_cutoff(
        source_coords=qm_coords,
        ext_coords=mm_coords,
        residues_array=residues_array,
        cutoff=mm_cut,
    )
    mm_top = cut_topology(top=mm_top, idx=mm_sel)

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
    residues_array = get_residues_array(
        topology=mm_topology, mm_indices=np.arange(mm_coords.shape[0])
    )
    num_pol, pol_coords, pol_idx, _ = whole_residues_cutoff(
        source_coords=qm_coords,
        ext_coords=mm_coords,
        residues_array=residues_array,
        cutoff=pol_cut,
    )
    pol_top = cut_topology(top=mm_topology, idx=pol_idx)
    return pol_coords, pol_top, pol_idx
