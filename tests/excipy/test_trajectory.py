import pytest
import numpy as np
import pytraj as pt
from excipy.trajectory import iterxyz, parse_ensure_order, parse_masks


# ============================================================================
# Helpers
# ============================================================================


def get_ala3_traj():
    fn, tn = pt.testing.get_fn("ala3")
    traj = pt.Trajectory(fn, top=tn)
    top = traj.top
    return traj, top


def get_parsed_ordered_unordered():
    """
    Use `parse_ensure_order` with different `names` to return coordinates
    and atomic numbers parsed in different order.
    """
    traj, top = get_ala3_traj()
    mask = ":1@H3,H2,N"
    # order as specified in mask
    names_ordmask = ["H3", "H2", "N"]
    # order as specified in top
    names_ordtop = ["N", "H2", "H3"]
    xyz_ordmask, z_ordmask = parse_ensure_order(
        traj, mask=mask, top=top, names=names_ordmask
    )
    xyz_ordtop, z_ordtop = parse_ensure_order(
        traj, mask=mask, top=top, names=names_ordtop
    )
    # mask order -> top order
    reorder_atoms = [2, 1, 0]
    return (xyz_ordmask, z_ordmask), (xyz_ordtop, z_ordtop), reorder_atoms


# ============================================================================
# Tests
# ============================================================================


def test_iterxyz():
    """
    Test that `iterxyz` gives the same coordinates as the batched function of
    `pytraj`.
    """
    traj, top = get_ala3_traj()
    mask = ":1"
    xyz = traj[mask].xyz.copy()
    res = iterxyz(traj, mask=mask)
    np.testing.assert_allclose(xyz, res)


def test_parse_ensure_order_xyz():
    """
    Test that different `names` in `parse_ensure_order` give different coordinates.
    """
    (
        (xyz_ordmask, z_ordmask),
        (xyz_ordtop, z_ordtop),
        reorder,
    ) = get_parsed_ordered_unordered()
    np.testing.assert_allclose(xyz_ordmask[:, reorder], xyz_ordtop)


def test_parse_ensure_order_z():
    """
    Test that different `names` in `parse_ensure_order` give different atomic numbers.
    """
    (
        (xyz_ordmask, z_ordmask),
        (xyz_ordtop, z_ordtop),
        reorder,
    ) = get_parsed_ordered_unordered()
    np.testing.assert_allclose(z_ordmask[reorder], z_ordtop)


def test_masks_not_list():
    """
    Test that `parse_masks` fails when `masks` is not given as a list.
    """
    traj, top = get_ala3_traj()
    masks = ":1@H3,H2"
    atom_names = ["H3", "H2"]
    with pytest.raises(ValueError):
        parse_masks(traj, masks=masks, atom_names=atom_names)


def test_atom_names_not_list():
    """
    Test that `parse_masks` fails when `atom_names` is not given as a list.
    """
    traj, top = get_ala3_traj()
    masks = [":1@H3,H2"]
    atom_names = "H3"
    with pytest.raises(ValueError):
        parse_masks(traj, masks=masks, atom_names=atom_names)


test_atom_names_not_list()
