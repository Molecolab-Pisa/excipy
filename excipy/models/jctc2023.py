from __future__ import annotations

from collections import defaultdict
from functools import partial

import numpy as np

from ..regression import (
    matern52_kernel,
    predict_gp,
    m52cm_linpot_kernel,
    LinearRidgeRegression,
)
from ..polar import mmpol_site_lr
from ..database import get_site_model_params, get_rescalings
from ..util import EV2CM, rescale_tresp, pbar

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
# * environment tresp
# * vacuum site energy
# * environment shift
# * environment site energy
# * polarizable LR contribution
# * polarizable site energy
#


def predict_vac_site_energy(encoding, model_params, residue_id, chl):
    # define a kernel functions that takes x and x_train only
    kernel_func = partial(
        matern52_kernel,
        lengthscale=model_params["lengthscale"],
        variance=model_params["variance"],
    )

    # define the key for caching the cholesky decomposition of k_train
    cache_key = (chl.upper(), "VAC")

    # mean and variance from gaussian process
    mean, variance = predict_gp(
        encoding,
        x_train=model_params["x_train"],
        kernel=kernel_func,
        mu=model_params["mu"],
        coefs=model_params["coefs"],
        sigma=model_params["sigma"],
        cache_key=cache_key,
    )

    return mean, variance


def predict_env_shift_site_energy(encoding, model_params, residue_id, chl):
    # define a kernel function that takes x and x_train only
    kernel_func = partial(
        m52cm_linpot_kernel,
        variance_lin=model_params["variance_lin"],
        variance_m52=model_params["variance_m52"],
        lengthscale_m52=model_params["lengthscale_m52"],
        cm_dims=model_params["cm_dims"],
        pot_dims=model_params["pot_dims"],
    )

    # define the key for caching the cholesky decomposition of k_train
    cache_key = (chl.upper(), "ENV_SHIFT")

    # mean and variance from gaussian process
    mean, variance = predict_gp(
        encoding,
        x_train=model_params["x_train"],
        kernel=kernel_func,
        mu=model_params["mu"],
        coefs=model_params["coefs"],
        sigma=model_params["sigma"],
        cache_key=cache_key,
    )

    return mean, variance


# specialize for chl a and b and bcl
# vac
predict_site_energies_cla_vac = partial(
    predict_vac_site_energy,
    chl="CLA",
)
predict_site_energies_chl_vac = partial(
    predict_vac_site_energy,
    chl="CHL",
)
predict_site_energies_bcl_vac = partial(
    predict_vac_site_energy,
    chl="BLC",
)
# env shift
predict_site_energies_cla_env_shift = partial(
    predict_env_shift_site_energy,
    chl="CLA",
)
predict_site_energies_chl_env_shift = partial(
    predict_env_shift_site_energy,
    chl="CHL",
)
predict_site_energies_bcl_env_shift = partial(
    predict_env_shift_site_energy,
    chl="BCL",
)


_PREDICT_SITE_ENERGIES_FUNCS = {
    ("CLA", "VAC"): predict_site_energies_cla_vac,
    ("CHL", "VAC"): predict_site_energies_chl_vac,
    ("BCL", "VAC"): predict_site_energies_bcl_vac,
    ("CLA", "ENV_SHIFT"): predict_site_energies_cla_env_shift,
    ("CHL", "ENV_SHIFT"): predict_site_energies_chl_env_shift,
    ("BCL", "ENV_SHIFT"): predict_site_energies_bcl_env_shift,
}

# wrapper


def predict_site_energies(encoding, type, model_params, residue_id, kind):
    """
    Predict the site energy of each molecule with Gaussian Process Regression.
    Arguments
    ---------
    encodings   : list of ndarray, (num_samples, num_features)
                Molecule encodings
    types       : list of str
                List of molecule types (used to fetch the model)
    residue_ids : list of str
                List of Residue IDs
    kind        : str
                Kind of site energy to predict (vac, env)
    Returns
    -------
    site energies : defaultdict
                  Dictionary with mean and variance of each
                  prediction.
    """

    iterator = pbar(
        encoding,
        total=encoding.shape[0],
        desc=": Predicting siten",
        ncols=79,
    )

    siten = defaultdict(list)

    func = _PREDICT_SITE_ENERGIES_FUNCS[(type.upper(), kind.upper())]

    for x in iterator:
        # shape (1, n_features)
        x = np.atleast_2d(x)

        mean, variance = func(x, model_params, residue_id)

        siten["y_mean"].append(mean)
        siten["y_var"].append(variance)

    for key in siten.keys():
        siten[key] = np.concatenate(siten[key], axis=0)

    return siten


def predict_tresp_charges(encodings, descriptors, types, residue_ids):
    """
    Predict the TrEsp charges of each molecule.
    Arguments
    ---------
    encodings    : list of ndarray, (num_samples, num_features)
                 Molecule encodings
    descriptors  : list of objects
                 Descriptors
    types        : list of str
                 List of molecule types
    residue_ids  : list of str
                 List of Residue IDs
    Returns
    -------
    tresp_charges : list of ndarray, (num_samples, num_atoms)
                  Predicted TrEsp charges
    """
    iterator = pbar(
        zip(encodings, descriptors, types, residue_ids),
        total=len(encodings),
        desc=": Predicting TrEsp",
        ncols=79,
    )
    tresp_charges = []
    for encoding, descriptor, type, residue_id in iterator:
        iterator.set_postfix(residue_id=f"{residue_id}")
        # Instantiate a regressor
        # The alpha parameter here is unimportant, as the regression
        # parameters are loaded from the database
        ridge = LinearRidgeRegression()
        ridge.load_params(type, "JCTC2023")
        # Note that the predicted charges have to be permuted
        # again in order to match the correct atom ordering
        y_permuted = ridge.predict(encoding)
        inverse_permutation = descriptor.permutator.inverse_transform
        y = inverse_permutation(y_permuted)
        tresp_charges.append(y)
    return tresp_charges


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
        scaling = get_rescalings([mol.type], "JCTC2023")[0]
        return rescale_tresp([mol.vac_tresp], [scaling])[0]

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
        return predict_site_energies(
            encoding, mol.type, params, mol.resid, kind="env_shift"
        )

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
