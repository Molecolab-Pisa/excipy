from __future__ import annotations
import functools

import numpy as np

from .util import read_molecule_types, get_dipoles, rescale_tresp
from .util import EV2CM
from .database import (
    get_atom_names,
    get_identical_atoms,
    select_masks,
    get_rescalings,
    get_hydrogens,
    get_site_model_params,
)
from .trajectory import parse_masks
from .descriptors import get_coulomb_matrix, encode_geometry, get_MM_elec_potential
from .regression import predict_tresp_charges, predict_site_energies
from .polar import mmpol_site_lr


# Implementation of the available models
#
# Each model is a collection of functions that takes as input a molecule
# object (see below). Each function uses properties of the molecule
# object with no checks on whether these properties are there (mixin).
# If you add more models, use the same pattern, and if you want to use a
# property that the Molecule object does not posses, implement that property.
# If all models are coherent, then in the CLI we can be agnostic about what
# they really do.
#
# In theory, each model should implement the following predictions:
#
# * vacuum tresp
# * electrostatic-embedding tresp (not present here)
# * polarizable-embedding tresp
# * vacuum site energy
# * environment shift
# * environment site energy
# * polarizable LR contribution
# * polarizable site energy
#


class Model_JCTC2023:
    """The same type of model published in

    [1] Cignoni, Edoardo, Lorenzo Cupellini, and Benedetta Mennucci.
        Journal of Chemical Theory and Computation 19.3 (2023).
    [2] Cignoni, Edoardo, Lorenzo Cupellini, and Benedetta Mennucci.
        Journal of Physics: Condensed Matter 34.30 (2022).

    OK for Chlorophyll a, b, and Bacteriochloropyhll a.
    """

    def __init__(self):
        pass

    @staticmethod
    def vac_tresp(mol):
        coulmat = mol.permuted_coulmat
        encoding = mol.permuted_coulmat_encoding
        return predict_tresp_charges([encoding], [coulmat], [mol.type], [mol.resid])[0]

    @staticmethod
    def pol_tresp(mol):
        return rescale_tresp([mol.vac_tresp], [mol.tresp_pol_scaling])[0]

    @staticmethod
    def vac_site_energy(mol):
        params = get_site_model_params(mol.type, kind="vac", model="JCTC2023")
        return predict_site_energies(
            mol.coulmat_noh_encoding, mol.type, params, mol.resid, kind="vac"
        )

    @staticmethod
    def env_shift_site_energy(mol):
        params = get_site_model_params(mol.type, kind="env", model="JCTC2023")
        encoding = np.column_stack(
            [mol.coulmat_noh_encoding, mol.elec_potential_encoding]
        )
        return predict_site_energies(encoding, mol.type, params, mol.resid, kind="env")

    @staticmethod
    def env_site_energy(mol):
        sites = dict()
        sites["y_mean"] = (
            mol.vac_site_energy["y_mean"] + mol.env_shift_site_energy["y_mean"]
        )
        sites["y_var"] = (
            mol.vac_site_energy["y_var"] + mol.env_shift_site_energy["y_var"]
        )
        return sites

    @staticmethod
    def pol_LR_site_energy(mol):
        lr, _ = mmpol_site_lr(
            mol.traj,
            coords=mol.coords,
            charges=mol.pol_tresp,
            residue_id=mol.resid,
            mask=mol.mask,
            pol_threshold=mol.pol_cutoff,
            db=mol.charges_db,
            mol2=mol.template_mol2,
            cut_strategy="spherical",
            smear_link=True,
            turnoff_mask=mol.turnoff_mask,
        )
        lr = lr.reshape(-1, 1)
        lr /= EV2CM
        sites = dict(y_mean=lr, y_var=np.zeros_like(lr))
        return sites

    @staticmethod
    def pol_site_energy(mol):
        sites = dict()
        sites["y_mean"] = (
            mol.vac_site_energy["y_mean"]
            + mol.env_shift_site_energy["y_mean"]
            + mol.pol_LR_site_energy["y_mean"]
        )
        sites["y_var"] = (
            mol.vac_site_energy["y_var"]
            + mol.env_shift_site_energy["y_var"]
            + mol.pol_LR_site_energy["y_var"]
        )
        return sites


# List of the available models per molecule type

available_models = {
    "CLA": {"JCTC2023": Model_JCTC2023()},
    "CHL": {"JCTC2023": Model_JCTC2023()},
    # the JCTC2023 model for BCL is called like this because it
    # has the same structure, just different parameters.
    "BCL": {"JCTC2023": Model_JCTC2023()},
}


class Molecule:
    def __init__(
        self,
        traj,
        resid,
        model_dict,
        elec_cutoff=30.0,
        pol_cutoff=20.0,
        turnoff_mask=None,
        charges_db=None,
        template_mol2=None,
    ):
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
        # tresp rescaling from vac to pol
        self.tresp_pol_scaling = get_rescalings([self.type])[0]
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

    def get_model(self, model_dict):
        try:
            model_str = model_dict[self.type]
        except KeyError as e:
            raise e(
                f"Cannot find a model for {self.type} in the given model dictionary."
            )
        return available_models[self.type][model_str]

    @property
    @functools.lru_cache()
    def coords(self):
        coords, atnums = parse_masks(self.traj, [self.mask], [self.atom_names])
        self.atnums = atnums[0]
        return coords[0]

    @property
    @functools.lru_cache()
    def coords_noh(self):
        coords, atnums = parse_masks(self.traj, [self.mask_noh], [self.atom_names_noh])
        self.atnums_noh = atnums[0]
        return coords[0]

    @property
    @functools.lru_cache()
    def permuted_coulmat(self):
        coulmat = get_coulomb_matrix(
            coords=[self.coords],
            atnums=[self.atnums],
            residue_ids=[self.resid],
            permute_groups=[self.permute_groups],
        )[0]
        return coulmat

    @property
    @functools.lru_cache()
    def permuted_coulmat_encoding(self):
        return encode_geometry([self.permuted_coulmat], free_attr=["coords"])[0]

    @property
    @functools.lru_cache()
    def coulmat_noh(self):
        coulmat = get_coulomb_matrix(
            coords=[self.coords_noh],
            atnums=[self.atnums_noh],
            residue_ids=[self.resid],
            permute_groups=None,
        )[0]
        return coulmat

    @property
    @functools.lru_cache()
    def coulmat_noh_encoding(self):
        return encode_geometry([self.coulmat_noh], free_attr=["coords"])[0]

    @property
    @functools.lru_cache()
    def elec_potential(self):
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
    def elec_potential_encoding(self):
        return encode_geometry([self.elec_potential])[0]

    @property
    @functools.lru_cache()
    def vac_tresp(self):
        return self.model.vac_tresp(self)

    @property
    @functools.lru_cache()
    def vac_tr_dipole(self):
        return get_dipoles([self.coords], [self.vac_tresp])[0]

    @property
    @functools.lru_cache()
    def pol_tresp(self):
        return self.model.pol_tresp(self)

    @property
    @functools.lru_cache()
    def pol_tr_dipole(self):
        return get_dipoles([self.coords], [self.pol_tresp])[0]

    @property
    @functools.lru_cache()
    def vac_site_energy(self):
        return self.model.vac_site_energy(self)

    @property
    @functools.lru_cache()
    def env_shift_site_energy(self):
        return self.model.env_shift_site_energy(self)

    @property
    @functools.lru_cache()
    def env_site_energy(self):
        return self.model.env_site_energy(self)

    @property
    @functools.lru_cache()
    def pol_LR_site_energy(self):
        return self.model.pol_LR_site_energy(self)

    @property
    @functools.lru_cache()
    def pol_site_energy(self):
        return self.model.pol_site_energy(self)
