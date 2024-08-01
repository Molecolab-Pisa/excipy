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
Collection of functions that read and write from/to the excipy database.
When including a new pigment, you must insert its data in the database.
See the available pigments to understand what data should be added.
Then, you can read the data inside the database using the functions defined
in this module.

A collection of functions is used to check whether parameters are available
in the database. For example ``atom_names_in_database`` will check if atom
names for a pigment are present

>>> atom_names_in_database('CLA') # chlorophyll a
True
>>> atom_names_in_database('CLL') # pigment not present
False

analogous functions check whether parameters are present (``params_in_database``), and so on.

Then, there are getter functions to get the parameters from the database.
For the case of the atom names

>>> cla_atom_names = get_atom_names('CLA')

You can also exclude some atoms via the ``exclude_atoms`` argument

>>> hydrogens = get_hydrogens('CLA')
>>> cla_atom_names_noh = get_atom_names('CLA', exclude_atoms=hydrogens)

Take a look at the module functions to see what parameters you can load.

Setter functions are present if you wish to register some variable inside
the database, for example for a new molecule

>>> set_atom_names('CLL', cll_atom_names)
"""
from __future__ import annotations
from typing import List, Union, Optional
import os
import json
from collections.abc import Iterable
import numpy as np


EXCIPY_DIR = os.path.dirname(__file__)
DATABASE_DIR = os.path.join(EXCIPY_DIR, "database")


# =============================================================================
# Custom class providing database paths
# =============================================================================


class DatabasePaths(object):
    """
    Class representing the excipy database

    Parameters
    ----------
    database_dir: str
        path to the excipy database folder.
    """

    def __init__(self, database_dir):
        self.database_dir = database_dir

    @property
    def atom_names(self):
        "path to the 'atom_names' folder inside the database."
        return os.path.join(self.database_dir, "atom_names")

    @property
    def params(self):
        "path to the 'parameters' folder inside the database."
        return os.path.join(self.database_dir, "parameters")

    @property
    def rescalings(self):
        "path to the 'rescalings' folder inside the database."
        return os.path.join(self.database_dir, "rescalings")

    @property
    def models(self):
        "path to the 'models' folder inside the database"
        return os.path.join(self.database_dir, "models")


database_paths = DatabasePaths(DATABASE_DIR)


def set_database_folder(folder: str) -> None:
    """sets the excipy database folder

    Arguments
    ---------
    folder: str
        path to the new database folder.
    """
    database_paths.database_dir = folder


# =============================================================================
# Custom exception
# =============================================================================


class DatabaseError(Exception):
    "generic Database exception"
    pass


# =============================================================================
# Checking functions
# =============================================================================


def atom_names_in_database(type: str) -> bool:
    """
    Checks that the .json file for type `type` is inside the
    database/atom_names.

    Arguments
    ---------
    type: str
        molecule type.
    """
    path = os.path.join(database_paths.atom_names, type + ".json")
    return os.path.isfile(path)


def params_in_database(type: str, model: str) -> bool:
    """
    Checks that the .json file for type `type` and the given
    `model` is inside the database/parameters.

    Arguments
    ---------
    type: str
        molecule type.
    model: str
        model string.
    """
    path = os.path.join(database_paths.params, model, type + ".json")
    return os.path.isfile(path)


def rescalings_in_database(type: str, model: str) -> bool:
    """
    Checks that the .json file for type `type` and the given
    `model` is inside the database/rescalings.

    Arguments
    ---------
    type: str
        molecule type.
    model: str
        model string.
    """
    path = os.path.join(database_paths.rescalings, model, type + ".json")
    return os.path.isfile(path)


# =============================================================================
# Getting functions
# =============================================================================


def _get_atom_names(type):
    if atom_names_in_database(type):
        path = os.path.join(database_paths.atom_names, type + ".json")
        return np.asarray(load_json(path)["names"].split())
    else:
        raise DatabaseError(
            f"atom names not present in database for molecule type {type}."
        )


def get_atom_names(type: str, exclude_atoms: Iterable[str] = None) -> np.ndarray:
    """
    Load the atom names from the database

    Parameters
    ---------
    type: str or list of str
        molecule type(s)
    exclude_atoms: list of str
        names of the atoms that will be excluded.
    """
    if isinstance(type, Iterable) and not isinstance(type, str):
        atom_names = [_get_atom_names(t) for t in type]
        if exclude_atoms is None:
            return atom_names
        else:
            atom_names = [
                np.asarray([n for n in names if n not in excluded])
                for names, excluded in zip(atom_names, exclude_atoms)
            ]
        return atom_names
    else:
        atom_names = _get_atom_names(type)
        if exclude_atoms is None:
            return atom_names
        else:
            atom_names = np.asarray([n for n in atom_names if n not in exclude_atoms])


def _get_atomic_numbers(type):
    if atom_names_in_database(type):
        path = os.path.join(database_paths.atom_names, type + ".json")
        return np.asarray(load_json(path)["atomic_numbers"]).astype(int)
    else:
        raise DatabaseError(
            f"atomic numbers not present in database for molecule type {type}."
        )


def get_atomic_numbers(type: str, exclude_atoms: Iterable[str] = None) -> np.ndarray:
    """
    Load the atomic numbers from the database

    Parameters
    ---------
    type: str or list of str
        molecule type(s)
    exclude_atoms: list of str
        names of the atoms that will be excluded.
    """
    if isinstance(type, Iterable) and not isinstance(type, str):
        atomic_numbers = np.asarray([_get_atomic_numbers(t) for t in type])
        if exclude_atoms is None:
            return atomic_numbers.astype(int)
        else:
            atomic_numbers = [
                np.asarray([n for n in names if n not in excluded])
                for names, excluded in zip(atomic_numbers, exclude_atoms)
            ]
        return atomic_numbers.astype(int)
    else:
        atomic_numbers = _get_atomic_numbers(type)
        if exclude_atoms is None:
            return atomic_numbers
        else:
            atomic_numbers = np.asarray(
                [n for n in atomic_numbers if n not in exclude_atoms]
            ).astype(int)


def _get_hydrogens(type):
    if atom_names_in_database(type):
        path = os.path.join(database_paths.atom_names, type + ".json")
        return np.asarray(load_json(path)["hydrogens"].split())
    else:
        raise DatabaseError(
            f"atom names not present in database for molecule type {type}."
        )


def get_hydrogens(type: str) -> np.ndarray:
    """
    Load the hydrogen atoms from the database

    Parameters
    ---------
    type: str or list of str
        molecule type(s)
    """
    if isinstance(type, Iterable) and not isinstance(type, str):
        return [_get_hydrogens(t) for t in type]
    else:
        return _get_hydrogens(type)


# def _get_ring_bonds(type, return_names=True):
#     if atom_names_in_database(type):
#         path = os.path.join(database_paths.atom_names, type + ".json")
#         bond_indices = np.asarray(load_json(path)["bonds"])
#         if return_names:
#             atom_names = get_atom_names(type)
#             bond_names = np.asarray(
#                 [[[atom_names[b[0]], atom_names[b[1]]] for b in bond_indices]]
#             )
#             return bond_names
#         else:
#             return bond_indices
#     else:
#         raise DatabaseError(
#             f"atom names not present in database for molecule type {type}."
#         )
#
#
# def get_ring_bonds(type):
#     """
#     Load the positions/atom names forming the bonds of the conjugated ring.
#     """
#     if isinstance(type, Iterable) and not isinstance(type, str):
#         return [_get_ring_bonds(t) for t in type]
#     else:
#         return _get_ring_bonds(type)


def _get_identical_atoms(
    type: str, convert_to_indeces: Optional[bool] = True
) -> List[Union[str, int]]:
    if atom_names_in_database(type):
        path = os.path.join(database_paths.atom_names, type + ".json")
        atom_names = load_json(path)["identical"]
        if not convert_to_indeces:
            return atom_names
        else:
            names = np.asarray(load_json(path)["names"].split())
            indeces = []
            for group in atom_names:
                group_indeces = []
                for atom in group:
                    pos = np.where(atom == names)[0]
                    group_indeces.append(pos)
                group_indeces = np.concatenate(group_indeces)
                indeces.append(group_indeces)
            return indeces
    else:
        raise DatabaseError(
            f"identical atoms not present in database for molecule type {type}."
        )


def get_identical_atoms(
    type: str, convert_to_indeces: Optional[bool] = True
) -> List[Union[str, int]]:
    """
    Load the regression parameters from the database.

    Parameters
    ---------
    type: str or list of str
        molecule type(s)
    convert_to_indices : bool
        whether to get the indeces of the identical atoms
        instead of the names.
    """
    if isinstance(type, Iterable) and not isinstance(type, str):
        return [
            _get_identical_atoms(t, convert_to_indeces=convert_to_indeces) for t in type
        ]
    else:
        return _get_identical_atoms(type, convert_to_indeces=convert_to_indeces)


def _get_params(type, model):
    if params_in_database(type, model):
        path = os.path.join(database_paths.params, model, type + ".json")
        return load_json(path)
    else:
        raise DatabaseError(
            f"params not present in database for molecule type {type}, model {model}"
        )


def get_params(type: str, model: str) -> np.ndarray:
    """
    Load the regression parameters from the database.

    Parameters
    ---------
    type: str or list of str
        molecule type(s)
    model: str
        model string.
    """
    if isinstance(type, Iterable) and not isinstance(type, str):
        return [_get_params(t, model) for t in type]
    else:
        return _get_params(type, model)


def _get_rescalings(type, model):
    if rescalings_in_database(type, model):
        path = os.path.join(database_paths.rescalings, model, type + ".json")
        return load_json(path)["mean"]
    else:
        raise DatabaseError(
            f"rescalings not present in database for molecule type {type}, model {model}."
        )


def get_rescalings(type: str, model: str) -> float:
    """
    Load the rescaling parameters for (vacuum TrEsp
    to environment TrEsp) from the database.

    Arguments
    ---------
    type: str or list of str
        molecule type(s).
    model: str
        model string.
    """
    if isinstance(type, Iterable) and not isinstance(type, str):
        return [_get_rescalings(t, model) for t in type]
    else:
        return _get_rescalings(type, model)


def get_site_model_params(type: str, kind: str, model: str) -> np.ndarray:
    """
    Get the site energy model parameters from the database.

    Arguments
    ---------
    type: str
        molecule type.
    kind: str
        kind of model ("vac" or "env").
    model: str
        model string.

    Returns
    -------
    params: np.ndarray
        model parameters.
    """
    if (kind == "vac") or (kind == "env"):
        path = os.path.join(
            database_paths.models, model, f"GPR_{kind}_{type}/model.npz"
        )
        try:
            return np.load(path)
        except FileNotFoundError:
            return DatabaseError(
                f"Model {model}/GPR_{kind}_{type} not found in database."
            )


# =============================================================================
# Setting functions
# =============================================================================


def set_atom_names(type, names, force_overwrite=False):
    if atom_names_in_database(type):
        if force_overwrite:
            pass
        else:
            raise DatabaseError(
                f"atom names for molecule type {type} already present and `force_overwrite` is False."
            )
    path = os.path.join(database_paths.atom_names, type + ".json")
    dump_json(path, names)


def set_params(type, model, params, force_overwrite=False):
    if params_in_database(type, model):
        if force_overwrite:
            pass
        else:
            raise DatabaseError(
                f"params for molecule type {type}, model {model} already present and `force_overwrite` is False."
            )
    path = os.path.join(database_paths.params, model, type + ".json")
    dump_json(path, params)


# =============================================================================
# Setup selection masks
# =============================================================================


def select_masks(residue_ids, atom_names, exclude_atoms=None):
    """
    Create an AMBER-like mask with the correct atoms
    """
    masks = []
    if exclude_atoms is None:
        exclude_atoms = [None] * len(residue_ids)
    for resid, names, excluded in zip(residue_ids, atom_names, exclude_atoms):
        if excluded is not None:
            names = [n for n in names if n not in excluded]
        mask = ":" + resid + "@" + ",".join(names)
        masks.append(mask)
    return masks


# =============================================================================
# Loading/dumping functions
# =============================================================================


def load_json(file):
    with open(file, "r") as f:
        return json.load(f)


def dump_json(file, obj):
    with open(file, "w+") as f:
        json.dump(obj, f)
