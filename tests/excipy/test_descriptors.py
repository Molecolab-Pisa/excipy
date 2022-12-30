import pytest
import numpy as np
import pytraj as pt
from excipy.descriptors import CoulombMatrix

# ============================================================================
# Helpers
# ============================================================================


def get_ala3_traj():
    fn, tn = pt.testing.get_fn("ala3")
    traj = pt.Trajectory(fn, top=tn)
    top = traj.top
    return traj, top


# ============================================================================
# Tests
# ============================================================================


@pytest.mark.parametrize("mask", [":1@H2,H3", ":1", ":1@N"])
@pytest.mark.parametrize("triu", [True, False])
def test_coulmat_shape(triu, mask):
    """
    Test that encoding a Coulomb Matrix with `triu=True` gives an encoding with
    the correct dimensionality.
    """
    traj, top = get_ala3_traj()
    n_atoms = top[mask].n_atoms
    coords = traj[mask].xyz.copy()
    atnums = np.array([atom.atomic_number for atom in top[mask].atoms])
    coul_mat = CoulombMatrix(
        coords=coords,
        atnums=atnums,
        residue_id="1",
        triu=triu,
        permute_groups=None,
    )
    encoding = coul_mat.encode()
    if triu is False:
        shape = (traj.n_frames, n_atoms, n_atoms)
    else:
        shape = (traj.n_frames, int(n_atoms * (n_atoms - 1) / 2.0))
    assert encoding.shape == shape
