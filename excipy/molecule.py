from __future__ import annotations
from typing import Dict, Union
import functools

import numpy as np

from .util import read_molecule_types, get_dipoles
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
        # residue type (CLA, CHL, ...)
        self.type = read_molecule_types(self.traj.top, [self.resid])[0]
        # get model
        self.model = self.get_model(model_dict)
        # load atom names from database
        self.atom_names = get_atom_names([self.type])[0]
        self.n_atoms = len(self.atom_names)
        # groups of identical atoms
        self.permute_groups = get_identical_atoms([self.type])[0]
        # amber mask
        self.mask = select_masks([self.resid], [self.atom_names])[0]
        # hydrogen atoms
        self.hydrogen_atoms = get_hydrogens([self.type])[0]
        # atom names without hydrogens
        self.atom_names_noh = get_atom_names(
            [self.type], exclude_atoms=[self.hydrogen_atoms]
        )[0]
        # mask without hydrogens
        self.mask_noh = select_masks([self.resid], [self.atom_names_noh])[0]
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

    def get_model(self, model_dict: Dict[str, str]) -> "Model":  # noqa: F821
        try:
            model_str = model_dict[self.type]
        except KeyError as e:
            raise e(
                f"Cannot find a model for {self.type} in the given model dictionary."
            )
        return available_models[self.type][model_str]

    @property
    @functools.lru_cache()
    def coords(self) -> np.ndarray:
        coords, atnums = parse_masks(self.traj, [self.mask], [self.atom_names])
        self.atnums = atnums[0]
        return coords[0]

    @property
    @functools.lru_cache()
    def coords_noh(self) -> np.ndarray:
        coords, atnums = parse_masks(self.traj, [self.mask_noh], [self.atom_names_noh])
        self.atnums_noh = atnums[0]
        return coords[0]

    @property
    @functools.lru_cache()
    def permuted_coulmat(self) -> "CoulombMatrix":  # noqa: F821
        coulmat = get_coulomb_matrix(
            coords=[self.coords],
            atnums=[self.atnums],
            residue_ids=[self.resid],
            permute_groups=[self.permute_groups],
        )[0]
        return coulmat

    @property
    @functools.lru_cache()
    def permuted_coulmat_encoding(self) -> np.ndarray:
        return encode_geometry([self.permuted_coulmat], free_attr=["coords"])[0]

    @property
    @functools.lru_cache()
    def coulmat_noh(self) -> "CoulombMatrix":  # noqa: F821
        coulmat = get_coulomb_matrix(
            coords=[self.coords_noh],
            atnums=[self.atnums_noh],
            residue_ids=[self.resid],
            permute_groups=None,
        )[0]
        return coulmat

    @property
    @functools.lru_cache()
    def coulmat_noh_encoding(self) -> np.ndarray:
        return encode_geometry([self.coulmat_noh], free_attr=["coords"])[0]

    @property
    @functools.lru_cache()
    def elec_potential(self) -> "MMElectrostaticPotential":  # noqa: F821
        return get_MM_elec_potential(
            traj=self.traj,
            masks=[self.mask],
            cutoff=self.elec_cutoff,
            frames=None,
            turnoff_mask=self.turnoff_mask,
            charges_db=self.charges_db,
            remove_mean=False,
        )[0]

    @property
    @functools.lru_cache()
    def elec_potential_encoding(self) -> np.ndarray:
        return encode_geometry([self.elec_potential])[0]

    @property
    @functools.lru_cache()
    def vac_tresp(self) -> np.ndarray:
        return self.model.vac_tresp(self)

    @property
    @functools.lru_cache()
    def vac_tr_dipole(self) -> np.ndarray:
        return get_dipoles([self.coords], [self.vac_tresp])[0]

    @property
    @functools.lru_cache()
    def pol_tresp(self) -> np.ndarray:
        return self.model.pol_tresp(self)

    @property
    @functools.lru_cache()
    def pol_tr_dipole(self) -> np.ndarray:
        return get_dipoles([self.coords], [self.pol_tresp])[0]

    @property
    @functools.lru_cache()
    def vac_site_energy(self) -> Dict[str, np.ndarray]:
        return self.model.vac_site_energy(self)

    @property
    @functools.lru_cache()
    def env_shift_site_energy(self) -> Dict[str, np.ndarray]:
        return self.model.env_shift_site_energy(self)

    @property
    @functools.lru_cache()
    def env_site_energy(self) -> Dict[str, np.ndarray]:
        return self.model.env_site_energy(self)

    @property
    @functools.lru_cache()
    def pol_LR_site_energy(self) -> Dict[str, np.ndarray]:
        return self.model.pol_LR_site_energy(self)

    @property
    @functools.lru_cache()
    def pol_site_energy(self) -> Dict[str, np.ndarray]:
        return self.model.pol_site_energy(self)
