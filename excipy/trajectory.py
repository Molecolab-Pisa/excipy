from collections.abc import Iterable
import numpy as np
import pytraj as pt
from .util import pbar
from .database import get_atomic_numbers


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


def parse_ensure_order(traj, mask, top, names, typ):
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
    traj_atom_numbs = get_atomic_numbers(typ)
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


def _validate_iterable_of_strings(arg, argname):
    basemsg = f"{argname} should be given as an iterable of strings."
    if isinstance(arg, str):
        raise ValueError(basemsg + f" Maybe you meant [{repr(arg)}]")
    elif not isinstance(arg, Iterable):
        raise ValueError(basemsg + f" You provided {type(arg)}")
    else:
        for elem in arg:
            if not isinstance(elem, str):
                raise ValueError(
                    basemsg + f" You provided an iterable of {type(elem)}."
                )


def _validate_masks(arg):
    return _validate_iterable_of_strings(arg, argname="masks")


def _validate_atom_names(arg):
    if isinstance(arg, Iterable) and not isinstance(arg, str):
        for elem in arg:
            _validate_iterable_of_strings(elem, argname="Elements of `atom_names`")
    else:
        raise ValueError(
            "atom_names should be given as a list of lists of str"
            + ", e.g., for a single residue, [['H1', 'H2']]"
        )


def parse_masks(traj, masks, atom_names, types):
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
               Each sublist if a list of atom names (str)
    Returns
    -------
    coords     : list of ndarray, (num_samples, num_atoms, 3)
               List of atomic coordinates
    atnums     : list of ndarray, (num_atoms,)
               List of atomic numbers
    """
    _validate_masks(masks)
    _validate_atom_names(atom_names)
    if len(masks) != len(atom_names):
        raise ValueError(
            "masks and atom_names should have the same length, but"
            + f" len(masks)={len(masks)} and len(atom_names)={len(atom_names)}"
        )

    coords = []
    atnums = []
    iterator = zip(masks, atom_names, types)
    for mask, names, typ in iterator:
        xyz, z = parse_ensure_order(traj, mask, traj.top, names, typ)
        coords.append(xyz)
        atnums.append(z)
    return coords, atnums
