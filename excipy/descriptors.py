# excipy: machine learning models for a fast estimation of excitonic Hamiltonians
# Copyright (C) 2022 excipy authors
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
#######
Summary
#######
Module where the molecular descriptors are implemented. Each descriptor
is implemented as a class that, when instantiated, takes all the necessary
arguments to encode the geometry. Then, the encoding is obtained by calling
the ``.encode()`` method of the class with no arguments.

Currently, the following descriptors are available:

* CoulombMatrix (with and without permutations)
* BondMatrix (experimental, currently not used)
* MMElectrostaticPotential

If you need to add another descriptor, please feel free to do it and follow
the same pattern.
"""
from __future__ import annotations
from typing import List, Any, Optional, Union
import numpy as np
import pytraj as pt
from collections.abc import Iterable
from .util import ANG2BOHR, pbar
from .elec import read_electrostatics
from .selection import get_residues_array, whole_residues_cutoff


# =============================================================================
# Encoding function (top level)
# =============================================================================


def encode_geometry(
    descriptors: List[Any], free_attr: Optional[str] = None
) -> List[np.ndarray]:
    """
    Encode the geometry with the given descriptors.

    Arguments
    ---------
    descriptors  : list of object
                 descriptor with an `.encode()` method
                 that encode the molecular geometry
    free_attr    : None or iterable
                 Attribute(s) of the descriptor to eliminate after
                 the encoding.

    Returns
    -------
    encodings    : list of ndarray, (num_samples, num_features)
                 Molecule encodings.
    """
    encodings = []
    for d in descriptors:
        encoding = d.encode()
        encodings.append(encoding)
        if free_attr is not None:
            if isinstance(free_attr, Iterable) and not isinstance(free_attr, str):
                for attr in free_attr:
                    if hasattr(d, attr):
                        delattr(d, attr)
                    else:
                        raise RuntimeError(
                            "Attribute {attr} not found in Descriptor {d}."
                        )
            else:
                raise RuntimeError(
                    "Cannot iterate free_attr (try providing a list or tuple)"
                )
    return encodings


# =============================================================================
# Functions obtaining descriptors
# =============================================================================


def _get_coulomb_matrix(coords, atnums, residue_id, triu=True, permute_groups=None):
    """
    Get the Coulomb Matrix associated to a single molecule.
    Arguments
    ---------
    coords          : ndarray, (num_samples, num_atoms, 3)
                    Atomic coordinates
    atnums          : ndarray, (num_atoms,)
                    Atomic numbers
    residue_ids     : str
                    Residue ID
    triu            : bool
                    Whether to encode only the upper triangular part (True) or
                    the whole matrix (False)
    permute_groups  : list of lists
                    Atoms that have to be permuted
    Returns
    -------
    CoulombMatrix   : excipy.descriptors.CoulombMatrix
                    A CoulombMatrix object
    """
    return CoulombMatrix(
        coords=coords,
        atnums=atnums,
        residue_id=residue_id,
        triu=triu,
        permute_groups=permute_groups,
    )


def get_coulomb_matrix(
    coords: List[np.ndarray],
    atnums: List[np.ndarray],
    residue_ids: List[str],
    triu: Optional[bool] = True,
    permute_groups: List[List[int]] = None,
) -> List[CoulombMatrix]:
    """
    Get the Coulomb Matrix associated with one or more molecule(s).

    Arguments
    ---------
    coords          : list of ndarray, (num_samples, num_atoms, 3)
                    List of atomic coordinates
    atnums          : list of ndarray, (num_atoms,)
                    List of atomic numbers
    residue_ids     : list of str
                    List of residue ID
    triu            : bool
                    Whether to encode only the upper triangular part (True) or
                    the whole matrix (False)
    permute_groups  : list of lists
                    Atoms that have to be permuted

    Returns
    -------
    CoulombMatrix   : list of excipy.descriptors.CoulombMatrix
                    CoulombMatrix object(s)
    """
    if permute_groups is None:
        permute_groups = [None] * len(coords)
    return [
        _get_coulomb_matrix(
            coords=c, atnums=a, residue_id=r, triu=triu, permute_groups=p
        )
        for c, a, r, p in zip(coords, atnums, residue_ids, permute_groups)
    ]


def _get_bond_matrix(coords, top, mask):
    """
    Get the bond matrix associated with a single molecule.
    Arguments
    ---------
    coords    : ndarray, (num_samples, num_atoms, 3)
              Atomic coordinates
    top       : pytraj.Topology
              Topology
    mask      : str
              AMBER mask
    Returns
    -------
    BondMatrix  : excipy.descriptors.BondMatrix
                BondMatrix object
    """
    return BondMatrix(
        coords=coords,
        top=top,
        mask=mask,
    )


def get_bond_matrix(
    coords: List[np.ndarray], top: pt.Topology, masks: List[str]
) -> List[BondMatrix]:
    """
    Get the bond matrix associated with one or more molecules.

    Arguments
    ---------
    coords   : list of ndarray, (num_samples, num_atoms, 3)
             List of atomic coordinates
    top      : pytraj.Topology
             Topology
    masks    : list of str
             List of AMBER masks

    Returns
    -------
    l        : list of excipy.descriptors.BondMatrix
             List of BondMatrix objects
    """
    return [_get_bond_matrix(coords=c, top=top, mask=m) for c, m in zip(coords, masks)]


def _get_MM_elec_potential(
    traj, mask, cutoff, frames, turnoff_mask, charges_db, remove_mean, read_alphas
):
    """
    Get the MM electrostatic potential acting on a single molecule.

    Arguments
    ---------
    traj         : pytraj.Trajectory
                 Trajectory
    mask         : str
                 AMBER mask
    cutoff       : float
                 Cutoff value for MM electrostatics
    frames       : None or ndarray
                 Frames to load
    turnoff_mask : str
                 AMBER mask of residues for which the electrostatics is not computed.

    Returns
    -------
    MMElectrostaticPotential : excipy.descriptors.MMElectrostaticPotential
                             MMElectrostaticPotential object
    """
    return MMElectrostaticPotential(
        traj=traj,
        mask=mask,
        cutoff=cutoff,
        frames=frames,
        turnoff_mask=turnoff_mask,
        charges_db=charges_db,
        remove_mean=remove_mean,
        read_alphas=read_alphas,
    )


def get_MM_elec_potential(
    traj: pt.TrajectoryIterator,
    masks: List[str],
    cutoff: float,
    frames: Union[None, np.ndarray],
    turnoff_mask: str,
    charges_db: Union[None, str],
    remove_mean: bool,
    read_alphas: bool,
) -> List[MMElectrostaticPotential]:
    """
    Get the MM electrostatic potential acting on one or more molecules.

    Arguments
    ---------
    traj         : pytraj.Trajectory
        Trajectory
    masks        : list of str
        List of AMBER mask
    cutoff       : float
        Cutoff value for MM electrostatics
    frames       : None or ndarray
        Frames to load
    turnoff_mask : str
        AMBER mask of residues for which the electrostatics is not computed.
    charges_db: str
        path to the charges database, if given.
    remove_mean: bool
        whether to remove the mean over the QM atoms from the potential.
    read_alphas: bool
        whether to read/infer the polarizabilities in addition to the charges.

    Returns
    -------
    l            : list of excipy.descriptors.MMElectrostaticPotential
                 MMElectrostaticPotential objects
    """
    return [
        _get_MM_elec_potential(
            traj=traj,
            mask=m,
            cutoff=cutoff,
            frames=frames,
            turnoff_mask=turnoff_mask,
            charges_db=charges_db,
            remove_mean=remove_mean,
            read_alphas=read_alphas,
        )
        for m in masks
    ]


# =============================================================================
# Descriptor Classes
# =============================================================================


class CoulombMatrix(object):
    """
    Coulomb Matrix encoder.
    In order to compute the Coulomb Matrix descriptor, you can do

    >>> cm = CoulombMatrix(coords, atnums, triu=False).encode()

    and this will give you the full Coulomb Matrix.
    If you need only the upper triangular part (no diagonal), then

    >>> cm = CoulombMatrix(coords, atnums, triu=True).encode()

    If you need to compute a permuted Coulomb Matrix, where identical
    atoms are reordered according to the L2 norm of the corresponding
    row, you must provide a list of lists, where each sublist is a list
    of identical atoms. For example, if atoms 0, 1, and 2 are identical

    >>> cm = CoulombMatrix(coords, atnums, triu=False,
    ...                    permute_groups=[[0, 1, 2]]).encode()

    Arguments
    ---------
    coords          : ndarray, (num_samples, num_atoms, 3)
                    Atomic coordinates
    atnums          : ndarray, (num_atoms,)
                    Atomic numbers
    residue_ids     : str
                    Residue ID (only used in progress bar info)
    triu            : bool
                    Whether to encode only the upper triangular part (True) or
                    the whole matrix (False)
    permute_groups  : list of lists
                    Atoms that have to be permuted
    """

    def __init__(self, coords, atnums, residue_id="", triu=True, permute_groups=None):
        self.coords = coords
        self.atnums = atnums
        self.residue_id = residue_id
        self.triu = triu
        self.permute_groups = permute_groups

    def _encode(self):
        """
        Encode the molecular geometry as a Coulomb matrix.
        """
        encoding = []

        qq = self.atnums[:, None] * self.atnums
        diag = 0.5 * self.atnums**2.4

        iterator = pbar(
            self.coords,
            desc=f": Encoding residue {self.residue_id}",
            ncols=79,
        )

        for xyz in iterator:
            dd = np.sum((xyz[:, None, :] - xyz) ** 2, axis=2) ** 0.5
            coul = np.divide(qq, dd, where=dd != 0)
            np.fill_diagonal(coul, diag)
            encoding.append(coul)

        return np.asarray(encoding)

    def _select_triu(self, encoding):
        """
        Select the upper triangular part of a list of matrices,
        excluding the diagonal.
        Arguments
        ---------
        encoding  : ndarray, (num_samples, num_atoms, num_atoms)
        """
        n_atoms = encoding.shape[1]
        encoding = [e[np.triu_indices(n_atoms, k=1)] for e in encoding]
        return np.concatenate([encoding])

    def encode(self) -> np.ndarray:
        """
        Encode the molecular geometry as a Coulomb matrix.
        """
        encoding = self._encode()

        if self.permute_groups is not None:
            # Store the matrix permutator as an attribute of the Coulomb Matrix
            # Needed as we have to access the permutator during prediction, to
            # perform the inverse permutation
            self.permutator = MatrixPermutator(permute_groups=self.permute_groups)
            self.permutator.fit(encoding)
            encoding = self.permutator.transform(encoding)

        if self.triu:
            encoding = self._select_triu(encoding)

        return encoding


class MatrixPermutator(object):
    """
    An object that permutes rows and columns of matrices, or entries of an array,
    based on the L2 norm of selected rows and columns of some matrices.

    Example
    -------
    If we have four atoms, C, C1, C2, C3, and C1, C2, C3 that are identical, the following
    two interaction matrices are equal

    >>> #     C  C1 C2 C3            C  C1 C2 C3
    >>> A = [[1, 1, 0, 0]      B = [[1, 0, 1, 0]
    ...      [1, 1, 0, 0]           [0, 1, 0, 0]
    ...      [0, 0, 1, 0]           [1, 0, 1, 0]
    ...      [0, 0, 0, 1]]          [0, 0, 0, 1]]

    It then makes sense to permute rows and columns of the two matrices so that they also appear as equal.
    Sorting according to the L2 norm, they become

    >>> #     C  C1 C2 C3            C  C1 C2 C3
    >>> A = [[1, 0, 0, 1]      B = [[1, 0, 0, 1]
    ...      [0, 1, 0, 0]           [0, 1, 0, 0]
    ...      [0, 0, 1, 0]           [0, 0, 1, 0]
    ...      [1, 0, 0, 1]]          [1, 0, 0, 1]]

    This class implements this transformation. It fits (stores the permuted indeces of the rows/
    columns to be permuted) a list of matrices, and can transform a list of matrices (the same
    or different ones) or a list of vectors.

    Transforming vectors can be useful as, if the permuted matrices are employed to fit a multiple-output
    regression model, one may also want to permute the target (otherwise the 1 to 1 correspondence between
    feature and target gets lost).

    For example, to permute the rows/columns of identical atoms in the
    CoulombMatrix

    >>> cm = CoulombMatrix(coords, atnums, triu=False).encode()
    >>> cm_permuted = MatrixPermutator(permute_groups).fit_transform(cm)

    Note that you don't have to do this manually, instead the
    ``MatrixPermutator`` class is handled inside the ``CoulombMatrix``
    class.

    The Matrix Permutator can also permute the positions in vectors,
    instead of matrices.

    Arguments
    ---------
    permute_groups : list of ndarrays
                   indeces of the rows/columns to be permuted.
    """

    def __init__(self, permute_groups):
        self.permute_groups = np.asarray(permute_groups)

    def fit(self, matrices: List[np.ndarray]):
        """
        Finds the permutation indices.

        Arguments
        ---------
        matrices : list of ndarrays (k,k), or ndarray (n_mats, k, k)
                 list of matrices to fit.
        """
        self.permuted_idx_ = []
        for mat in matrices:
            permutation = []
            for pg in self.permute_groups:
                norm = np.linalg.norm(mat[pg], axis=1)
                argsort = np.argsort(norm)
                permutation.append(pg[argsort])
            self.permuted_idx_.append(np.asarray(permutation))
        self.permuted_idx_ = np.asarray(self.permuted_idx_)
        return self

    def transform(
        self, arrays: np.ndarray, subset_idxs: Optional[np.ndarray] = None
    ) -> List[np.ndarray]:
        """
        Direct (forward) transformation: from a list of unpermuted arrays, generates
        a list of permuted arrays.

        Arguments
        ---------
        arrays   : list or ndarray
                 arrays to be transformed (row/column-permuted).
        subset_idxs : None or ndarray (num_idxs,)
                 subset of indeces from the full vector of fitted indeces
                 used to perform the transformation.
                 Example:
                 If the transformation is carried out on a training set only,
                 defined by a vector train_idx storing the indeces of the train
                 set (e.g., [0, 2, 3]), then train_idx can be passed as subset_idx,
                 and only the permutations corresponding to these indeces are used
                 to transform the arrays.

        Returns
        -------
        trasnformed_arrays : list or ndarray
                           transformed arrays
        """

        # self._check_arrays(arrays)

        # Create a copy of the arrays to be modified inplace
        arrays = np.asarray(arrays.copy())

        # Decide upone whether you have to transform vectors or matrices
        transform_fn = self._choose_transform_fn(arrays)

        return transform_fn(arrays, mode="forward", subset_idxs=subset_idxs)

    def fit_transform(
        self, arrays: np.ndarray, subset_idxs: Optional[np.ndarray] = None
    ) -> List[np.ndarray]:
        """
        Run `fit` and `transform` on the same arrays
        """
        return self.fit(arrays).transform(arrays, subset_idxs=subset_idxs)

    def inverse_transform(
        self, arrays: np.ndarray, subset_idxs: Optional[np.ndarray] = None
    ) -> List[np.ndarray]:
        """
        Inverse (backward) transformation: from a list of permuted arrays, restores
        the original list of arrays.

        Arguments
        ---------
        arrays   : list or ndarray
                 arrays to be transformed (row/column-permuted).
        subset_idxs : None or ndarray (num_idxs,)
                 subset of indeces from the full vector of fitted indeces
                 used to perform the transformation.
                 Example:
                 If the transformation is carried out on a training set only,
                 defined by a vector train_idx storing the indeces of the train
                 set (e.g., [0, 2, 3]), then train_idx can be passed as subset_idx,
                 and only the permutations corresponding to these indeces are used
                 to transform the arrays.

        Returns
        -------
        trasnformed_arrays : list or ndarray
                           transformed arrays
        """

        # self._check_arrays(arrays)

        # Create a copy of the arrays to be modified inplace
        arrays = np.asarray(arrays.copy())

        # Decide upone whether you have to transform vectors or matrices
        transform_fn = self._choose_transform_fn(arrays)

        return transform_fn(arrays, mode="backward", subset_idxs=subset_idxs)

    # def _check_arrays(self, arrays):
    #     num_arrays_to_transform = len(arrays)
    #     num_arrays_fitted = len(self.permuted_idx_)
    #     if num_arrays_fitted != num_arrays_to_transform:
    #         raise RuntimeError('Number of fitted arrays is different from the number of arrays to be transformed')

    def _choose_transform_fn(self, arrays):
        """
        Choose the transformation function based on the shape of the provided
        arrays. `arrays` should arrive here as a ndarray of shape (num_arrays, k)
        if each array is a vector, or (num_arrays, k, k) if each array is a matrix.
        """
        shape = arrays.shape
        if len(shape) == 2:
            return self._transform_vectors
        elif len(shape) == 3:
            return self._transform_matrices
        else:
            raise RuntimeError("Only vectors and matrices are supported.")

    def _select_from_mode(self, mode):
        """
        Select the initial and final indeces from the required mode.
        mode == 'forward' specifies a transformation from unpermuted indeces to permuted ones.
        mode == 'backward' specifies a transformation from permtued indeces to unpermuted ones.
        """
        permuted = self.permuted_idx_
        # Same shape as permuted
        unpermuted = self.permute_groups[None, :].repeat(len(permuted), axis=0)
        if mode == "forward":
            transform_to = permuted
            transform_from = unpermuted
        elif mode == "backward":
            transform_to = unpermuted
            transform_from = permuted
        else:
            # The user should not be able to specify 'forward' or 'backward' directly
            raise RuntimeError("You should not be here.")
        return transform_from, transform_to

    def _transform_vectors(self, vecs, mode="forward", subset_idxs=None):
        """
        Transformation for 1D arrays.
        """
        transform_from, transform_to = self._select_from_mode(mode)
        if subset_idxs is not None:
            transform_from, transform_to = (
                transform_from[subset_idxs],
                transform_to[subset_idxs],
            )
        for vec, old_idxs, new_idxs in zip(vecs, transform_from, transform_to):
            for old_idx, new_idx in zip(old_idxs, new_idxs):
                vec[old_idx] = vec[new_idx]
        return vecs

    def _transform_matrices(self, mats, mode="forward", subset_idxs=None):
        """
        Transformation for 1D arrays.
        """
        transform_from, transform_to = self._select_from_mode(mode)
        if subset_idxs is not None:
            transform_from, transform_to = (
                transform_from[subset_idxs],
                transform_to[subset_idxs],
            )
        for mat, old_idxs, new_idxs in zip(mats, transform_from, transform_to):
            for old_idx, new_idx in zip(old_idxs, new_idxs):
                mat[:, old_idx] = mat[:, new_idx]
                mat[old_idx, :] = mat[new_idx, :]
        return mats


class BondMatrix(object):
    """bond matrix descriptor

    Encode the geometry as a list of covalent bond values.

    Parameters
    ----------
    coords   : ndarray, (num_samples, num_atoms, 3)
             Atomic coordinates
    top      : pytraj.Topology
             Topology
    mask     : str
             AMBER mask
    """

    def __init__(self, coords, top, mask):
        self.coords = coords
        self.top = top
        self.mask = mask
        self.mask_atoms = self.get_mask_atoms(mask)
        self.residue_id = self.get_mask_resid(mask)

    def get_mask_atoms(self, mask):
        atoms = mask.split("@")[1].split(",")
        atoms = np.asarray(atoms)
        return atoms

    def get_mask_resid(self, mask):
        resid = mask.split("@")[0][1:]
        return resid

    def select_atoms(self):
        return self.top.select(self.mask)

    def select_bonds(self, atoms):
        return [
            (i, j) for (i, j) in self.top.bond_indices if (i in atoms and j in atoms)
        ]

    def read_bond_names(self, bonds):
        atom = self.top.atom
        return np.asarray([(atom(i).name, atom(j).name) for (i, j) in bonds])

    def get_bond_indices(self, names):
        indices = []
        for pname in names:
            idx1 = np.where(pname[0] == self.mask_atoms)[0][0]
            idx2 = np.where(pname[1] == self.mask_atoms)[0][0]
            indices.append([idx1, idx2])
        indices = np.asarray(indices)
        return indices

    def calc_bond_lengths(self, indices):
        bond_lengths = []

        iterator = pbar(
            self.coords,
            desc=f": Encoding residue {self.residue_id}",
            ncols=79,
        )

        for xyz in iterator:
            # xyz[indices] is [n_bond_lengths, 2, 3]
            bl = np.sum(np.diff(xyz[indices], axis=1)[:, 0] ** 2, axis=-1) ** 0.5
            bond_lengths.append(bl)

        return np.asarray(bond_lengths)

    def _encode(self):
        atoms = self.select_atoms()
        bonds = self.select_bonds(atoms)
        names = self.read_bond_names(bonds)
        # Ensure that the ordering of atom bonds
        # strictly follows the atom order given in the
        # mask (order with which coordinates are loaded)
        indices = self.get_bond_indices(names)
        return self.calc_bond_lengths(indices)

    def encode(self):
        return self._encode()


class MMElectrostaticPotential(object):
    """electrostatic potential descriptor

    Electrostatic potential of a MM trajectory.
    To compute the electrostatic potential of the MM part on the QM
    atoms, with an electrostatic cutoff of 30 angstrom, you can do

    >>> pot = MMElectrostaticPotential(
    ...     traj=traj,
    ...     mask=mask,
    ...     cutoff=30.,
    ... ).encode()

    You can decide to switch off some MM residue by providing the
    corresponding mask. For example, to turn off residue 1

    >>> pot = MMElectrostaticPotential(
    ...     traj=traj,
    ...     mask=mask,
    ...     cutoff=30.,
    ...     turnoff_mask=":10",
    ... ).encode()

    If you want to remove the mean of the potential over the QM atoms

    >>> pot = MMElectrostaticPotential(
    ...     traj=traj,
    ...     mask=mask,
    ...     cutoff=30.,
    ...     remove_mean=True
    ... ).encode()
    >>> print(np.mean(pot, axis=1)) # will print 0

    If you don't want to use the prmtop charges, you can provide a
    database file. It is a three column file with the following
    columns: RESIDUE_NAME (str) ATOM_NAME (str) CHARGE (float).
    Then, give it to the class

    >>> pot = MMElectrostaticPotential(
    ...     traj=traj,
    ...     mask=mask,
    ...     cutoff=30.,
    ...     charges_db=charges_db,
    ... ).encode()

    Arguments
    ---------
    traj       : pytraj.Trajectory
    mask       : str
               Amber mask of the atoms where you want to compute the potential
    cutoff     : float
               Environment cutoff (atoms more distant than cutoff are excluded from the environment)
    frames     : None or list
               Frames considered for the calculation of the potential
    turnoff_mask : str
                 Mask of the atoms that you do not want to include in the environment
    charges_db  : str
                Path to a charges database.
                The database is assumed to be of (at least) three columns
                RESIDUE_NAME   ATOM_NAME   CHARGE
                example:

                ACE   H1    0.112300
                ACE   CH3  -0.366200
                ...
                PC    O22  -0.596900
                Additional columns are ignored.
    remove_mean: bool
                whether to remove the mean over the QM atoms from the potential.
    read_alphas: bool
                whether to read/infer the polarizabilities in addition to the charges.
    """

    def __init__(
        self,
        traj: pt.TrajectoryIterator,
        mask: str,
        cutoff: float,
        frames: Optional[np.ndarray] = None,
        turnoff_mask: Optional[str] = None,
        charges_db: Optional[str] = None,
        remove_mean: Optional[bool] = False,
        read_alphas: Optional[bool] = True,
    ) -> None:
        self.traj = traj
        self.top = self.traj.top
        self.qm_mask = mask
        self.mm_mask = "!" + self.qm_mask
        self.turnoff_mask = turnoff_mask
        if turnoff_mask is not None:
            self.mm_mask = "(" + self.mm_mask + ")&(!" + turnoff_mask + ")"
        self.resid = mask.split("@")[0][1:]
        self.cutoff = cutoff
        self.frames = frames
        self.charges_db = charges_db
        self.remove_mean = remove_mean
        self.read_alphas = read_alphas

    def electrostatic_potential(
        self,
        coords1: np.ndarray,
        coords2: np.ndarray,
        charges2: np.ndarray,
        residues_array: np.ndarray,
    ) -> np.ndarray:
        """
        Computes the electrostatic potential for a single frame.

        Arguments
        ---------
        coords1: np.ndarray
            coordinates of the QM atoms
        coords2: np.ndarray
            coordinates of the MM atoms
        charges2: np.ndarray
            charges of the MM atoms
        residues_array: np.ndarray
            residues_array of the MM atoms
            (see excipy.selection.get_residues_array)

        Returns
        -------
        pot: np.ndarray
            electrostatic potential on the QM atoms.
        """
        _, coords2, mask, dd = whole_residues_cutoff(
            source_coords=coords1,
            ext_coords=coords2,
            residues_array=residues_array,
            cutoff=self.cutoff,
        )
        # dd = np.sum((coords1[:, None] - coords2) ** 2, axis=2) ** 0.5
        pot = np.sum(charges2[mask, None] / dd, axis=0)
        return pot

    def calc_elec_potential_along_traj(
        self,
        qm_indices: np.ndarray,
        mm_indices: np.ndarray,
        mm_charges: np.ndarray,
        residues_array: np.ndarray,
    ) -> np.ndarray:
        """
        Computes the electrostatic potential on the QM atoms along a trajectory.

        Arguments
        ---------
        qm_indices: np.ndarray
            indices of the QM atoms.
        mm_indices: np.ndarray
            indices of the MM atoms.
        mm_charges: np.ndarray
            charges of the MM atoms.
        residues_array: np.ndarray
            residues_array of the MM atoms (see excipy.selection.get_residues_array).

        Returns
        -------
        pots: np.ndarray
            electrostatic potentials computed along the trajectory.
        """
        potentials = []
        total = self.traj.n_frames if self.frames is None else len(self.frames)

        iterator = pbar(
            pt.iterframe(self.traj, frame_indices=self.frames),
            desc=f": Encoding residue {self.resid}",
            total=total,
            ncols=79,
        )

        for frame in iterator:
            qm_coords = frame.xyz[qm_indices]  # Angstrom
            mm_coords = frame.xyz[mm_indices]  # Angstrom
            pot = self.electrostatic_potential(
                qm_coords,
                mm_coords,
                mm_charges,
                residues_array,
            )  # e / Angstrom
            pot = pot / ANG2BOHR  # e / Bohr
            potentials.append(pot)
        potentials = np.asarray(potentials)

        if self.remove_mean:
            potentials -= np.mean(potentials, axis=1)[:, None]

        return potentials

    def get_charges(self, indices: np.ndarray) -> np.ndarray:
        """
        Gets the charges for the MM part

        Arguments
        ---------
        indices: np.ndarray
            indices of the atoms of the MM part.

        Returns
        -------
        charges: np.ndarray
            charges of the MM part.
        """
        charges, _, _ = read_electrostatics(
            top=self.top,
            db=self.charges_db,
            mol2=None,
            warn=False,
            read_alphas=self.read_alphas,
        )
        return charges[indices]

    def _encode(self):
        qm_indices = self.top.select(self.qm_mask)
        mm_indices = self.top.select(self.mm_mask)
        mm_charges = self.get_charges(mm_indices)
        residues_array = get_residues_array(self.top, mm_indices)
        return self.calc_elec_potential_along_traj(
            qm_indices,
            mm_indices,
            mm_charges,
            residues_array,
        )

    def encode(self):
        "Encode the molecular geometry as the MM Electrostatic Potential"
        return self._encode()
