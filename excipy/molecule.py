from __future__ import annotations
from typing import Dict, Union
import functools

import numpy as np

from .util import read_molecule_types
from .database import (
    get_atom_names,
    get_identical_atoms,
    select_masks,
    get_hydrogens,
)
from .trajectory import parse_masks
from .descriptors import (
    get_coulomb_matrix,
    encode_geometry,
    get_MM_elec_potential,
)
from .models import available_models


def cached_with_checked_dependency(*dependencies):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            func_key = getattr(self, "_{:s}_cache_key".format(func.__name__), None)
            # cache not set (first call)
            if func_key is None:
                res = func(self, *args, **kwargs)
                new_key = [
                    getattr(self, "_{:s}_cache_key".format(dep)) for dep in dependencies
                ]
                setattr(self, "_{:s}_cache_key".format(func.__name__), new_key)
                setattr(self, "_{:s}".format(func.__name__), res)
                return res
            else:
                # check if function key matches with dependencies keys
                same_key = True
                for dep, subkey in zip(dependencies, func_key):  # need to check lengths
                    dep_key = getattr(self, "_{:s}_cache_key".format(dep), None)
                    # we are using lists, so == checks for equality of
                    # all elements (works with keys that are themselves lists)
                    check = dep_key == subkey
                    if not check:
                        same_key = False
                # no match, recompute and update cache key
                if not same_key:
                    res = func(self, *args, **kwargs)
                    new_key = [
                        getattr(self, "_{:s}_cache_key".format(dep))
                        for dep in dependencies
                    ]
                    setattr(self, "_{:s}_cache_key".format(func.__name__), new_key)
                    setattr(self, "_{:s}".format(func.__name__), res)
                    return res
                # match, return existing attribute
                else:
                    return getattr(self, "_{:s}".format(func.__name__))
            return res

        return wrapper

    return decorator


class Molecule:
    def __init__(
        self,
        traj: "pytraj.TrajectoryIterator",  # noqa: F821
        resid: Union[str, int],
        model_dict: Dict[str, str],
        elec_cutoff: float = 30.0,
        pol_cutoff: float = 20.0,
        turnoff_mask: str = None,
        charges_db: str = None,
        template_mol2: str = None,
        read_alphas: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        traj: pt.Trajectory
            pytraj TrajectoryIterator
        resid: str or int
            resid number
        model_dict: dict
            dictionary mapping the molecule type with the model name
        elec_cutoff: float
            cutoff for the static electrostatics potential descriptor
        pol_cutoff: float
            cutoff for the polarizable part of electrostatics
        turnoff_mask: str
            AMBER mask. Atoms matching the mask are excluded from the
            electrostatics (static and polarizable)
        charges_db: str
            path to the charges database file, useful to use different
            sets of charges when accounting for polarization.
        template_mol2: str
            path to the template mol2 file, which should be used together
            (but it's not necessary) with the charges database, in order
            to e.g. distinguish terminal residues (not identifiable from
            the prmtop).
        """
        self.traj = traj
        self.resid = str(resid)
        # get model
        self.model = self.get_model(model_dict)
        # electrostatics cutoff
        self.elec_cutoff = elec_cutoff
        # polarization cutoff
        self.pol_cutoff = pol_cutoff
        # turnoff mask
        self.turnoff_mask = turnoff_mask
        # charges database (for e.g. pol charges)
        self.charges_db = charges_db
        # tempalte mol2 (e.g. to recognize terminal residues)
        self.template_mol2 = template_mol2
        # read_alphas
        self.read_alphas = read_alphas

    @property
    def traj(self):
        return self._traj

    @traj.setter
    def traj(self, v):
        self._traj_cache_key = str(id(v))
        self._traj = v

    @property
    def resid(self):
        return self._resid

    @resid.setter
    def resid(self, v):
        self._resid_cache_key = str(v)
        self._resid = str(v)

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, v):
        self._model_cache_key = str(id(v))
        self._model = v

    @property
    def elec_cutoff(self):
        return self._elec_cutoff

    @elec_cutoff.setter
    def elec_cutoff(self, v):
        self._elec_cutoff_cache_key = str(v)
        self._elec_cutoff = v

    @property
    def pol_cutoff(self):
        return self._pol_cutoff

    @pol_cutoff.setter
    def pol_cutoff(self, v):
        self._pol_cutoff_cache_key = str(v)
        self._pol_cutoff = v

    @property
    def turnoff_mask(self):
        return self._turnoff_mask

    @turnoff_mask.setter
    def turnoff_mask(self, v):
        self._turnoff_mask_cache_key = str(v)
        self._turnoff_mask = v

    @property
    def charges_db(self):
        return self._charges_db

    @charges_db.setter
    def charges_db(self, v):
        self._charges_db_cache_key = str(v)
        self._charges_db = v

    @property
    def template_mol2(self):
        return self._template_mol2

    @template_mol2.setter
    def template_mol2(self, v):
        self._template_mol2_cache_key = str(v)
        self._template_mol2 = v

    @property
    def read_alphas(self):
        return self._read_alphas

    @read_alphas.setter
    def read_alphas(self, v):
        self._read_alphas_cache_key = str(v)
        self._read_alphas = v

    #
    # Derived/read-only properties
    #

    @property
    def type(self):
        return read_molecule_types(self.traj.top, [self.resid])[0]

    @property
    def atom_names(self):
        return get_atom_names([self.type])[0]

    @property
    def n_atoms(self):
        return len(self.atom_names)

    @property
    def permute_groups(self):
        return get_identical_atoms([self.type])[0]

    @property
    def mask(self):
        return select_masks([self.resid], [self.atom_names])[0]

    @property
    def hydrogen_atoms(self):
        return get_hydrogens([self.type])[0]

    @property
    def atom_names_noh(self):
        return get_atom_names([self.type], exclude_atoms=[self.hydrogen_atoms])[0]

    @property
    def mask_noh(self):
        return select_masks([self.resid], [self.atom_names_noh])[0]

    def get_model(self, model_dict: Dict[str, str]) -> "Model":  # noqa: F821
        try:
            model_str = model_dict[self.type]
        except KeyError as e:
            raise e(
                f"Cannot find a model for {self.type} in the given model dictionary."
            )
        return available_models[self.type][model_str]

    # Molecular properties that are independent of the ML model, and that can be
    # used by the ML models to compute the various quantities of interest
    #
    @property
    @cached_with_checked_dependency("traj", "resid")
    def coords(self) -> np.ndarray:
        "molecular coordinates"
        coords, atnums = parse_masks(
            self.traj, [self.mask], [self.atom_names], [self.type]
        )
        self.atnums = atnums[0]
        return coords[0]

    @property
    @cached_with_checked_dependency("traj", "resid")
    def coords_noh(self) -> np.ndarray:
        "molecular coordinates with no hydrogens"
        coords, atnums = parse_masks(
            self.traj, [self.mask_noh], [self.atom_names_noh], [self.type]
        )
        self.atnums_noh = atnums[0]
        return coords[0]

    @property
    @cached_with_checked_dependency("traj", "resid")
    def permuted_coulmat(self) -> "CoulombMatrix":  # noqa: F821
        "permuted (for identical atoms only) coulomb matrix"
        coulmat = get_coulomb_matrix(
            coords=[self.coords],
            atnums=[self.atnums],
            residue_ids=[self.resid],
            permute_groups=[self.permute_groups],
        )[0]
        return coulmat

    @property
    @cached_with_checked_dependency("traj", "resid")
    def permuted_coulmat_encoding(self) -> np.ndarray:
        "permuted (for identical atoms only) coulomb matrix encoding"
        return encode_geometry([self.permuted_coulmat], free_attr=["coords"])[0]

    @property
    @cached_with_checked_dependency("traj", "resid")
    def coulmat_noh(self) -> "CoulombMatrix":  # noqa: F821
        "coulomb matrix (offdiagonal) with no hydrogens"
        coulmat = get_coulomb_matrix(
            coords=[self.coords_noh],
            atnums=[self.atnums_noh],
            residue_ids=[self.resid],
            permute_groups=None,
        )[0]
        return coulmat

    @property
    @cached_with_checked_dependency("traj", "resid")
    def coulmat_noh_encoding(self) -> np.ndarray:
        "coulomb matrix (offdiagonal) with no hydrogens encoding"
        return encode_geometry([self.coulmat_noh], free_attr=["coords"])[0]

    @property
    @cached_with_checked_dependency(
        "traj",
        "resid",
        "elec_cutoff",
        "turnoff_mask",
        "charges_db",
        "template_mol2",
        "read_alphas",
    )
    def elec_potential(self) -> "MMElectrostaticPotential":  # noqa: F821
        "electrostatic potential on QM (or ML) atoms"
        return get_MM_elec_potential(
            traj=self.traj,
            masks=[self.mask],
            cutoff=self.elec_cutoff,
            frames=None,
            turnoff_mask=self.turnoff_mask,
            charges_db=self.charges_db,
            remove_mean=False,
            read_alphas=self.read_alphas,
        )[0]

    @property
    @cached_with_checked_dependency(
        "traj", "resid", "elec_cutoff", "turnoff_mask", "charges_db", "template_mol2"
    )
    def elec_potential_encoding(self) -> np.ndarray:
        "electrostatic potential on QM (or ML) atoms encoding"
        return encode_geometry([self.elec_potential])[0]

    # Quantities that are predicted by the ML models
    # All these properties are implemented using "delegation"
    # i.e. the calculation of each of these properties is
    # delegated to the chosen ML model.
    #
    # If you require further properties, simply add them, but
    # remember also to add the corresponding method to the Model
    # class for consistency.
    #
    # Each property must return a prediction for interface
    # consistency. The prediction has a value and an associated
    # uncertainty, or variance.
    #
    @property
    @cached_with_checked_dependency("traj", "resid", "model")
    def vac_tresp(self) -> "Prediction":  # noqa: F821
        "vacuum tresp charges"
        return self.model.vac_tresp(self)

    @property
    @cached_with_checked_dependency("traj", "resid", "model")
    def vac_tr_dipole(self) -> "Prediction":  # noqa: F821
        "vacuum transition dipole"
        return self.model.vac_tr_dipole(self)

    @property
    @cached_with_checked_dependency(
        "traj",
        "resid",
        "model",
        "elec_cutoff",
        "turnoff_mask",
        "charges_db",
        "template_mol2",
        "pol_cutoff",
        "read_alphas",
    )
    def env_tresp(self) -> "Prediction":  # noqa: F821
        "polarizable embedding tresp charges"
        return self.model.env_tresp(self)

    @property
    @cached_with_checked_dependency(
        "traj",
        "resid",
        "model",
        "elec_cutoff",
        "turnoff_mask",
        "charges_db",
        "template_mol2",
        "pol_cutoff",
        "read_alphas",
    )
    def env_tr_dipole(self) -> "Prediction":  # noqa: F821
        "polarizable embedding transition dipole"
        return self.model.env_tr_dipole(self)

    @property
    @cached_with_checked_dependency("traj", "resid", "model")
    def vac_site_energy(self) -> "Prediction":  # noqa: F821
        "vacuum site energy"
        return self.model.vac_site_energy(self)

    @property
    @cached_with_checked_dependency(
        "traj",
        "resid",
        "model",
        "elec_cutoff",
        "turnoff_mask",
        "charges_db",
        "template_mol2",
        "pol_cutoff",
        "read_alphas",
    )
    def env_shift_site_energy(self) -> "Prediction":  # noqa: F821
        "environment electrochromic shift"
        return self.model.env_shift_site_energy(self)

    @property
    @cached_with_checked_dependency(
        "traj",
        "resid",
        "model",
        "elec_cutoff",
        "turnoff_mask",
        "charges_db",
        "template_mol2",
        "pol_cutoff",
        "read_alphas",
    )
    def env_site_energy(self) -> "Prediction":  # noqa: F821
        "environment site energy"
        return self.model.env_site_energy(self)

    @property
    @cached_with_checked_dependency(
        "traj",
        "resid",
        "model",
        "elec_cutoff",
        "turnoff_mask",
        "charges_db",
        "template_mol2",
        "pol_cutoff",
    )
    def pol_LR_site_energy(self) -> "Prediction":  # noqa: F821
        "polarizable embedding Linear Response contribution"
        return self.model.pol_LR_site_energy(self)

    @property
    @cached_with_checked_dependency(
        "traj",
        "resid",
        "model",
        "elec_cutoff",
        "turnoff_mask",
        "charges_db",
        "template_mol2",
        "pol_cutoff",
    )
    def pol_site_energy(self) -> "Prediction":  # noqa: F821
        "polarizable embedding site energy"
        return self.model.pol_site_energy(self)
