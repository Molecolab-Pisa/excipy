import os
import json
from collections.abc import Iterable
import numpy as np
import tensorflow as tf


EXCIPY_DIR = os.path.dirname(__file__)
DATABASE_DIR = os.path.join(EXCIPY_DIR, "database")


# =============================================================================
# Custom class providing database paths
# =============================================================================


class DatabasePaths(object):
    def __init__(self, database_dir):
        self.database_dir = database_dir

    @property
    def atom_names(self):
        return os.path.join(self.database_dir, "atom_names")

    @property
    def params(self):
        return os.path.join(self.database_dir, "parameters")

    @property
    def rescalings(self):
        return os.path.join(self.database_dir, "rescalings")

    @property
    def models(self):
        return os.path.join(self.database_dir, "models")


database_paths = DatabasePaths(DATABASE_DIR)


def set_database_folder(folder):
    database_paths.database_dir = folder


# =============================================================================
# Custom exception
# =============================================================================


class DatabaseError(Exception):
    pass


# =============================================================================
# Checking functions
# =============================================================================


def atom_names_in_database(type):
    path = os.path.join(database_paths.atom_names, type + ".json")
    return os.path.isfile(path)


def params_in_database(type):
    path = os.path.join(database_paths.params, type + ".json")
    return os.path.isfile(path)


def rescalings_in_database(type):
    path = os.path.join(database_paths.rescalings, type + ".json")
    return os.path.isfile(path)


def type_in_database(type):
    return atom_names_in_database(type) and params_in_database(type)


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


def get_atom_names(type, exclude_atoms=None):
    """
    Load the atom names from the database
    Arguments
    ---------
    type    : str or list of str
            molecule type(s)
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


def _get_hydrogens(type):
    if atom_names_in_database(type):
        path = os.path.join(database_paths.atom_names, type + ".json")
        return np.asarray(load_json(path)["hydrogens"].split())
    else:
        raise DatabaseError(
            f"atom names not present in database for molecule type {type}."
        )


def get_hydrogens(type):
    """
    Load the hydrogen atoms from the database
    Arguments
    ---------
    type    : str or list of str
            molecule type(s)
    """
    if isinstance(type, Iterable) and not isinstance(type, str):
        return [_get_hydrogens(t) for t in type]
    else:
        return _get_hydrogens(type)


def _get_ring_bonds(type, return_names=True):
    if atom_names_in_database(type):
        path = os.path.join(database_paths.atom_names, type + ".json")
        bond_indices = np.asarray(load_json(path)["bonds"])
        if return_names:
            atom_names = get_atom_names(type)
            bond_names = np.asarray(
                [[[atom_names[b[0]], atom_names[b[1]]] for b in bond_indices]]
            )
            return bond_names
        else:
            return bond_indices
    else:
        raise DatabaseError(
            f"atom names not present in database for molecule type {type}."
        )


def get_ring_bonds(type):
    """
    Load the positions/atom names forming the bonds of the conjugated ring.
    """
    if isinstance(type, Iterable) and not isinstance(type, str):
        return [_get_ring_bonds(t) for t in type]
    else:
        return _get_ring_bonds(type)


def _get_identical_atoms(type, convert_to_indeces=True):
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


def get_identical_atoms(type, convert_to_indeces=True):
    """
    Load the identical atoms from the database.
    Arguments
    ---------
    type               : str or list of str
                       molecule type(s)
    convert_to_indeces : bool
                       whether to get the indeces of the identical
                       atoms instead of the names
    """
    if isinstance(type, Iterable) and not isinstance(type, str):
        return [
            _get_identical_atoms(t, convert_to_indeces=convert_to_indeces) for t in type
        ]
    else:
        return _get_identical_atoms(type, convert_to_indeces=convert_to_indeces)


def _get_params(type):
    if params_in_database(type):
        path = os.path.join(database_paths.params, type + ".json")
        return load_json(path)
    else:
        raise DatabaseError(f"params not present in database for molecule type {type}.")


def get_params(type):
    """
    Load the regression parameters from the database.
    Arguments
    ---------
    type    : str or list of str
            molecule type(s)
    """
    if isinstance(type, Iterable) and not isinstance(type, str):
        return [_get_params(t) for t in type]
    else:
        return _get_params(type)


def _get_rescalings(type):
    if rescalings_in_database(type):
        path = os.path.join(database_paths.rescalings, type + ".json")
        return load_json(path)["mean"]
    else:
        raise DatabaseError(
            f"rescalings not present in database for molecule type {type}."
        )


def get_rescalings(type):
    """
    Load the rescaling parameters for (vacuum TrEsp
    to environment TrEsp) from the database.
    """
    if isinstance(type, Iterable) and not isinstance(type, str):
        return [_get_rescalings(t) for t in type]
    else:
        return _get_rescalings(type)


def get_site_model(type, kind):
    """
    Get the site energy model from the database.
    """
    if (kind == "vac") or (kind == "env"):
        path = os.path.join(database_paths.models, f"GPR_{kind}_{type}")
        try:
            return tf.saved_model.load(path)
        except OSError:
            raise DatabaseError(f"Model GPR_{kind}_{type} not found in database.")
    else:
        raise RuntimeError(
            f"kind={kind} argument not recognized. Can't retrieve model from database."
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


def set_params(type, params, force_overwrite=False):
    if params_in_database(type):
        if force_overwrite:
            pass
        else:
            raise DatabaseError(
                f"params for molecule type {type} already present and `force_overwrite` is False."
            )
    path = os.path.join(database_paths.params, type + ".json")
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
