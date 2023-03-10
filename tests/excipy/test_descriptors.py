import pytest
import numpy as np
import pytraj as pt
from excipy.descriptors import CoulombMatrix, MatrixPermutator

# ============================================================================
# Helpers
# ============================================================================


def get_ala3_traj():
    fn, tn = pt.testing.get_fn("ala3")
    traj = pt.Trajectory(fn, top=tn)
    top = traj.top
    return traj, top


def ref_coulomb_matrix():
    """
    Simple reference Coulomb Matrix
    """
    coords = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    # do not put two equal numbers here or you may break
    # some test that assume different numbers here
    atnums = np.array([6.0, 1.0])
    encoding = np.zeros((2, 2))
    n_atoms = coords.shape[0]
    for i in range(n_atoms):
        for j in range(n_atoms):
            if i == j:
                encoding[i, i] = 0.5 * atnums[i] ** 2.4
            else:
                r_ij = np.sum((coords[i] - coords[j]) ** 2) ** 0.5
                encoding[i, j] = (atnums[i] * atnums[j]) / r_ij
    return coords, atnums, encoding


# ============================================================================
# Tests
# ============================================================================


@pytest.mark.parametrize("mask", [":1@H2,H3", ":1", ":1@N"])
@pytest.mark.parametrize("triu", [True, False])
def test_coulmat_shape(triu, mask):
    """
    Test that encoding a CoulombMatrix with `triu=True` gives an encoding with
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


def test_coulmat_output():
    """
    Test for the correct output of the CoulombMatrix encoding.
    """
    coords, atnums, ref_encoding = ref_coulomb_matrix()
    encoding = CoulombMatrix(
        coords=coords[None, :, :],
        atnums=atnums,
        residue_id="1",
        triu=False,
        permute_groups=None,
    ).encode()
    np.testing.assert_allclose(encoding[0], ref_encoding)


def test_coulmat_permutation():
    """
    Test that a CoulombMatrix with `permute_groups` different from
    `None` gives the correct output.
    """
    coords, atnums, ref_encoding = ref_coulomb_matrix()
    encoding = CoulombMatrix(
        coords=coords[None, :, :],
        atnums=atnums,
        residue_id="1",
        triu=False,
        permute_groups=np.array([[0, 1]]),
    ).encode()
    if atnums[0] < atnums[1]:
        pass
    elif atnums[0] > atnums[1]:
        encoding = encoding[:, ::-1]
        encoding = encoding[:, :, ::-1]
    else:
        assert False
    np.testing.assert_allclose(encoding[0], ref_encoding)


def test_permutator_forward():
    """
    Test that a MatrixPermutator yields the correct output.
    """
    A = np.array(
        [
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    B = np.array(
        [
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    C = np.array(
        [
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 1.0],
        ]
    )
    mp = MatrixPermutator(permute_groups=np.array([[1, 2, 3]])).fit([A, B])
    permuted = mp.transform([A, B])
    for matrix in permuted:
        np.testing.assert_equal(matrix, C)

    vA = np.array([1.0, 1.0, 0.0, 0.0])
    vB = np.array([1.0, 0.0, 1.0, 0.0])
    vC = np.array([1.0, 0.0, 0.0, 1.0])
    v_permuted = mp.transform([vA, vB])
    for vector in v_permuted:
        np.testing.assert_equal(vector, vC)


def test_permutator_backward():
    """
    Test that a MatrixPermutator can recover the original matrices
    from the permuted ones.
    """
    A = np.array(
        [
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    B = np.array(
        [
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    mp = MatrixPermutator(permute_groups=np.array([[1, 2, 3]])).fit([A, B])
    permuted = mp.transform([A, B])
    A_recovered, B_recovered = mp.inverse_transform(permuted)
    np.testing.assert_equal(A_recovered, A)
    np.testing.assert_equal(B_recovered, B)

    vA = np.array([1.0, 1.0, 0.0, 0.0])
    vB = np.array([1.0, 0.0, 1.0, 0.0])
    v_permuted = mp.transform([vA, vB])
    vA_recovered, vB_recovered = mp.inverse_transform(v_permuted)
    np.testing.assert_equal(vA_recovered, vA)
    np.testing.assert_equal(vB_recovered, vB)
