from collections.abc import Iterable
import numpy as np
import pytraj as pt
from .util import pbar


def iterxyz(traj, mask):
    """
    Iterate over a pytraj Trajectory and collect the atomic coordinates
    that correspond to the AMBER mask.
    Arguments
    ---------
    traj       : pytraj.Trajectory
               Trajectory
    mask       : str
               AMBER mask
    Returns
    -------
    coords     : ndarray, (num_samples, num_atoms, 3)
               Atomic coordinates
    """
    residue_id = mask[1:].split("@")[0]
    iterator = pbar(
        pt.iterframe(traj, mask=mask),
        desc=f": Loading residue {residue_id}",
        total=traj.n_frames,
        ncols=79,
    )
    # Needed to store a copy of the coordinates or it does not work.
    return np.asarray([frame.xyz.copy() for frame in iterator])


def parse_ensure_order(traj, mask, top, names):
    """
    Collect atomic coordinates and atomic numbers for atoms selected by an AMBER
    mask.
    Coordinates are collected matching the atom ordering specified in names.
    Arguments
    ---------
    traj      : pytraj.Trajectory
              Trajectory
    mask      : str
              AMBER mask
    top       : pytraj.Topology
              Topology
    names     : list of str
              List of atom names
    Returns
    -------
    xyz       : ndarray, (num_samples, num_atoms, 3)
              Atomic coordinates
    z         : ndarray, (num_atoms,)
              Atomic numbers
    """
    # Get the list of atom names that are parsed by pytraj
    # and the corresponding cartesian coordinates and atomic
    # numbers
    traj_atom_names = np.asarray([a.name for a in top[mask]])
    traj_atom_numbs = np.asarray([a.atomic_number for a in top[mask]])
    traj_xyz = iterxyz(traj, mask)
    # Empty containers for cartesian coordinates and atomic numbers
    xyz = np.zeros(traj_xyz.shape)
    z = []
    # Ensure that atom coordinates and atomic numbers are loaded
    # in the same order as the list of atoms specified in the database.
    for i, name in enumerate(names):
        atom_idx = np.where(name == traj_atom_names)[0]
        atom_xyz = np.squeeze(traj_xyz[:, atom_idx], axis=1)
        # Update the xyz with the atom coordinates
        xyz[:, i, :] = atom_xyz
        # Update the array of atomic numbers
        z.append(traj_atom_numbs[atom_idx])
    z = np.concatenate(z)
    return xyz, z


def _validate_masks(masks):
    basemsg = "masks should be given as an iterable of strings."
    if isinstance(masks, str):
        raise ValueError(basemsg + f" Maybe you meant [{repr(masks)}]")
    elif not isinstance(masks, Iterable):
        raise ValueError(basemsg + f" You provided {type(masks)}")


def parse_masks(traj, masks, atom_names):
    """
    Parse the AMBER masks, collecting coordinates and atomic numbers
    with the same atomic order as specified by atom_names.
    Arguments
    ---------
    traj       : pytraj.Trajectory
               Trajectory
    masks      : list of str
               List of AMBER masks
    atom_names : list of lists
               Each sublist if a list of atom names
    Returns
    -------
    coords     : list of ndarray, (num_samples, num_atoms, 3)
               List of atomic coordinates
    atnums     : list of ndarray, (num_atoms,)
               List of atomic numbers
    """
    _validate_masks(masks)
    coords = []
    atnums = []
    iterator = zip(masks, atom_names)
    for mask, names in iterator:
        xyz, z = parse_ensure_order(traj, mask, traj.top, names)
        coords.append(xyz)
        atnums.append(z)
    return coords, atnums
