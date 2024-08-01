#!/usr/bin/env python
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

import sys
import numpy as np
import pytraj as pt
from argparse import ArgumentParser
from collections import defaultdict

import excipy
from excipy.database import set_database_folder
from excipy.elec import compute_coulomb_couplings
from excipy.util import (
    create_hdf5_outfile,
    save_n_frames,
    save_residue_ids,
    save_coords,
    save_atom_numbers,
    save_tresp_charges,
    save_dipoles,
    save_coulomb_couplings,
    save_site_energies,
    load_n_frames,
    load_as_exat,
    dump_exat_files,
    Colors,
    block_average,
)

from excipy.molecule import Molecule

if excipy.available_polarizable_module:
    from excipy.polar import batch_mmpol_coup_lr


def logo():
    print(
        Colors.bold
        + Colors.green
        + """
    8888888888 Y88b   d88P  .d8888b. 8888888 8888888b. Y88b   d88P
    888         Y88b d88P  d88P  Y88b  888   888   Y88b Y88b d88P
    888          Y88o88P   888    888  888   888    888  Y88o88P
    8888888       Y888P    888         888   888   d88P   Y888P
    888           d888b    888         888   8888888P"     888
    888          d88888b   888    888  888   888           888
    888         d88P Y88b  Y88b  d88P  888   888           888
    8888888888 d88P   Y88b  "Y8888P" 8888888 888           888
    """
        + Colors.null
    )


def version():
    print(
        Colors.bold
        + Colors.green
        + "version: {:s}".format(excipy.__version__)
        + Colors.null
    )


def cli_parse(argv):
    "Command line parser"
    parser = ArgumentParser(
        prog=sys.argv[0],
        description="Machine learning models for a fast estimation of excitonic Hamiltonians.",
    )

    required = parser.add_argument_group("required")
    opt = required.add_argument

    opt(
        "-c",
        "--coordinates",
        required=True,
        help="Molecular Dynamics trajectory coordinates (.nc, .xtc, ...)",
    )

    opt(
        "-p",
        "--parameters",
        required=True,
        help="Molecular Dynamics parameters/topology (.prmtop, .gro, ...)",
    )

    opt(
        "-r",
        "--residue_ids",
        required=True,
        nargs="+",
        help="Residue IDs (e.g., 1 to select the first residue)",
    )

    optional = parser.add_argument_group("optional")
    opt = optional.add_argument

    opt(
        "-f",
        "--frames",
        required=False,
        nargs="+",
        default=[None, None, None],
        help="Frames to load, [start, stop, step] format, one-based count.",
    )

    opt(
        "-t",
        "--turnoff_mask",
        required=False,
        default=None,
        help="Amber mask to turn off (no electrostatics) some part of the MM region (e.g., ':1' to turn off electrostatics of resid 1)",
    )

    opt(
        "--cutoff",
        required=False,
        default=999.0,
        type=float,
        help="Distance cutoff in Angstrom (no coupling for molecules farther than cutoff). NOTE: if a list of coupling is provided by the --coup_list option, the cutoff is ignored.",
    )

    opt(
        "--coup_list",
        required=False,
        default=None,
        # type=list,
        nargs="+",
        help="Residues pairs you want to compute the couplings on, e.g. 664_665, 664_666. Has the priority on --cutoff option.",
    )

    opt(
        "--env_coup_threshold",
        required=False,
        default=1,
        type=float,
        help="Threshold for environment (both EE and Pol) couplings to be computed. Couplings below threshold are left as computed in vac.",
    )

    opt(
        "--no_coup",
        required=False,
        action="store_true",
        help="Whether to skip computing Coulomb couplings between chlorophylls.",
    )

    opt(
        "--no_siten",
        required=False,
        action="store_true",
        help="Whether to skip computing site energies between chlorophylls.",
    )

    opt(
        "--pol_cutoff",
        required=False,
        default=15.0,
        type=float,
        help="Polarization cutoff in Angstrom (no polarization for molecules farther than cutoff).",
    )

    opt(
        "--elec_cutoff",
        required=False,
        default=30.0,
        type=float,
        help="Cutoff for electrostatics in Angstrom.",
    )

    opt(
        "--no_coup_pol",
        required=False,
        action="store_true",
        help="Whether to not compute the MMPol contribution to the Coulomb coupling.",
    )

    opt(
        "--no_site_pol",
        required=False,
        action="store_true",
        help="Whether to not compute the MMPol contribution to the site energies.",
    )

    opt(
        "--no_site_env",
        required=False,
        action="store_true",
        help="Whether to not compute the chlorophyll site energy in environment.",
    )

    opt(
        "--models",
        required=False,
        metavar="KEY=VALUE",
        nargs="+",
        help="Specify the desired model for each molecule.",
    )

    opt(
        "--database_folder",
        required=False,
        default=None,
        help="Absolute path to a custom database folder.",
    )

    opt(
        "--charges_db",
        required=False,
        default=None,
        help="database file containing static and pol charges and polarizabilities",
    )

    outfiles = parser.add_argument_group("output files")
    opt = outfiles.add_argument

    opt(
        "--outfile",
        required=False,
        default="out.h5",
        help="Output file (HDF5).",
    )

    args = parser.parse_args(argv[1:])
    return args, parser


def parse_frames(frames, n_frames, return_slice=False):
    # Process frames
    if len(frames) > 3:
        errmsg = (
            f'"frames" field can take up to 3 arguments, you provided {len(frames)}. '
        )
        errmsg += 'Please provide "frames" in this format: [start, stop, step] (one-based count).'
        raise RuntimeError(errmsg)
    # Convert to zero-based counting
    start = int(frames[0]) - 1 if frames[0] is not None else 0
    if start < 0:
        raise RuntimeError(f"Start={start}. Maybe you used zero-based counting?")
    if frames[1] == "last" or frames[1] is None:
        stop = n_frames
    else:
        stop = int(frames[1])
    step = int(frames[2]) if frames[2] is not None else 1
    if return_slice:
        return (start, stop, step)
    else:
        return np.arange(start, stop, step)


def parse_models(models):
    model_dict = dict()
    if models:
        for key_value in models:
            key, value = key_value.split("=")
            key = key.strip().upper()
            model_dict[key] = value.upper()
    else:
        # provide a default
        model_dict["CLA"] = "JCTC2023"
        model_dict["CHL"] = "JCTC2023"
        model_dict["BCL"] = "JCTC2023"
    return model_dict


# =============================================================================
# Printing utilities
# =============================================================================


def print_cli_arguments(args):
    print("\n: " + "=" * 78)
    title = Colors.bold + Colors.green + "Command Line Arguments" + Colors.null
    print(": " + title)
    print(": " + "=" * 78)
    for arg in vars(args):
        fmt_arg = Colors.bold + Colors.orange + arg + Colors.null
        print(": {:40s} = {:35s}".format(fmt_arg, str(getattr(args, arg))))


def print_action(action_str):
    fmt_str = Colors.bold + Colors.magenta + action_str + Colors.null
    print("\n: " + fmt_str + " " + "=" * (78 - len(action_str) - 1) + "\n")


def print_section(section_str):
    fmt_str = Colors.bold + Colors.blue + section_str + Colors.null
    print("\n: " + fmt_str + " " + "=" * (78 - len(section_str) - 1) + "\n")


def print_molecules(types, residue_ids, masks):
    print()
    for type, residue_id, mask in zip(types, residue_ids, masks):
        print(f": Type={type}   Residue ID={residue_id}   N. Atoms={mask}")


def print_predicted_couplings(couplings, pairs, kind="V"):
    print_action("Predicted couplings")
    if pairs is None:
        return
    print(
        ": {:^20s} {:^10s} ({:^10s} {:^10s})  \n:".format(
            "Coupling", "Mean", "Low", "Up"
        )
    )
    for coup, pair in zip(couplings, pairs):
        coup_str = f"{kind}_{pair[0]}.{pair[1]}"
        coup_mean = np.mean(coup)
        coup_low = np.quantile(coup, 0.05)
        coup_up = np.quantile(coup, 0.95)
        print(
            ": {:20s} {:10.4f} ({:^10.4f} {:^10.4f})     cm^-1".format(
                coup_str,
                coup_mean,
                coup_low,
                coup_up,
            )
        )
    # print(":")


def print_predicted_sitens(sitens, residue_ids, kind="E"):
    print_action("Predicted Site Energies")
    print(
        ": {:^20s} {:^10s} ({:^10s} {:^10s})  \n:".format(
            "Site Energy",
            "Mean",
            "Low",
            "Up",
        )
    )
    for siten, resid in zip(sitens, residue_ids):
        siten_str = f"{kind}_{resid}"
        siten_mean = np.mean(siten)
        siten_low = np.quantile(siten, 0.05)
        siten_up = np.quantile(siten, 0.95)
        print(
            ": {:^20s} {:^10.4f} ({:^10.4f} {:^10.4f})      eV".format(
                siten_str,
                siten_mean,
                siten_low,
                siten_up,
            )
        )


def print_begin():
    title = Colors.bold + Colors.green + "BEGIN" + Colors.null
    print(f"\n: === {title} " + "=" * 68 + "\n")


def print_end():
    title = Colors.bold + Colors.green + "END" + Colors.null
    print(f"\n: === {title} " + "=" * 70 + "\n")


# =============================================================================
# Actions
# =============================================================================


def compute_couplings(molecules, args):  # noqa: C901
    print_section("COULOMB COUPLINGS")

    n_molecules = len(molecules)

    # Predict tresp charges
    print_action("Predicting vacuum TrEsp charges")
    vac_tresp = [mol.vac_tresp for mol in molecules]
    for p, m in zip(vac_tresp, molecules):
        save_tresp_charges(
            p.value,
            p.var,
            m.resid,
            kind="vac",
            outfile=args.outfile,
        )

    # Predicted vacuum transition dipoles
    vac_tr_dipoles = [mol.vac_tr_dipole for mol in molecules]
    for p, m in zip(vac_tr_dipoles, molecules):
        save_dipoles(p.value, p.var, m.resid, kind="vac", outfile=args.outfile)

    # Do not compute couplings if there is only one molecule
    if n_molecules < 2:
        pass
    else:
        print_action("Computing vacuum couplings")
        # Compute the couplings using the predicted TrEsp charges
        coords = [mol.coords for mol in molecules]

        vac_tresp_val = [p.value for p in vac_tresp]
        predicted_couplings, pairs_ids = compute_coulomb_couplings(
            coords, vac_tresp_val, args.residue_ids, args.cutoff, args.coup_list
        )

        print_predicted_couplings(predicted_couplings, pairs_ids, kind="V_vac")
        # No coupling to save if no molecule is within the cutoff distance
        if predicted_couplings is None:
            pass
        else:
            save_coulomb_couplings(
                predicted_couplings, pairs_ids, kind="vac", outfile=args.outfile
            )

    # Predict TrEsp charges in polarizable environment
    print_action("Rescaling TrEsp and Coulomb couplings")
    env_tresp = [mol.env_tresp for mol in molecules]
    for p, m in zip(env_tresp, molecules):
        save_tresp_charges(
            p.value,
            p.var,
            m.resid,
            kind="env",
            outfile=args.outfile,
        )

    # Predict transition dipoles in polarizable environment
    env_tr_dipoles = [mol.env_tr_dipole for mol in molecules]
    for p, m in zip(env_tr_dipoles, molecules):
        save_dipoles(
            p.value,
            p.var,
            m.resid,
            kind="env",
            outfile=args.outfile,
        )

    # Do not compute couplings if there is only one molecule
    if n_molecules < 2:
        pass
    else:
        # If no coupling has been computed, there's nothing to rescale
        if predicted_couplings is None:
            pass
        else:
            # Identify couplings above a chosen threshold (by taking the max value on the traj)
            # We will compute env and pol couplings only for these latter

            # Select the couplings above the threshold in at least one frame
            max_coup = abs(np.array(predicted_couplings)).max(axis=1)
            threshold_mask = np.where(max_coup > args.env_coup_threshold)[0]

            # Create a list of couplings above threshold to rescale
            above_threshold_couplings = [
                predicted_couplings[pos] for pos in threshold_mask
            ]

            # Keep trace of the pair ids we selected
            above_threshold_pairs_ids = [pairs_ids[pos] for pos in threshold_mask]

            # Create a list of the pair indeces above the threshold
            above_threshold_pairs_idx = []
            for p in above_threshold_pairs_ids:
                above_threshold_pairs_idx.append(
                    (args.residue_ids.index(p[0]), args.residue_ids.index(p[1]))
                )

            # update the coup list to contain only the residues
            above_threshold_coup_list = ["_".join(p) for p in above_threshold_pairs_ids]

            env_tresp_val = [p.value for p in env_tresp]
            env_couplings, _ = compute_coulomb_couplings(
                coords,
                env_tresp_val,
                args.residue_ids,
                args.cutoff,
                above_threshold_coup_list,
            )

            print_predicted_couplings(
                env_couplings, above_threshold_pairs_ids, kind="V_env"
            )
            save_coulomb_couplings(
                env_couplings,
                above_threshold_pairs_ids,
                kind="env",
                outfile=args.outfile,
            )

    # We compute the MMPol contribution to each coupling
    # if we can make use of the polarization module
    if excipy.available_polarizable_module and not args.no_coup_pol:
        if n_molecules < 2:
            pass
        # We compute this coupling only if we have
        # computed some couplings before
        elif above_threshold_couplings is None:
            pass
        else:
            # We compute this coupling only for those
            # pairs for which we have computed the environment coupling

            # pairs_idx = select_molecules(args.residue_ids, pairs_ids)

            print_action("Computing MMPol contribution")
            # MMPol contributions
            masks = [mol.mask for mol in molecules]
            mmpol_couplings, _ = batch_mmpol_coup_lr(
                molecules[0].traj,  # not nice but ok
                coords=coords,
                charges=env_tresp_val,
                residue_ids=args.residue_ids,
                masks=masks,
                pairs=above_threshold_pairs_idx,
                pol_threshold=args.pol_cutoff,
                db=args.charges_db,
                mol2=None,
                cut_strategy="spherical",
                smear_link=True,
                turnoff_mask=args.turnoff_mask,
            )
            print_predicted_couplings(
                mmpol_couplings, above_threshold_pairs_ids, kind="V_mmpol"
            )
            save_coulomb_couplings(
                mmpol_couplings,
                above_threshold_pairs_ids,
                kind="pol_shift",
                outfile=args.outfile,
            )

            # We also output the total coupling
            tot_couplings = []
            for coul, mmp in zip(env_couplings, mmpol_couplings):
                tot_couplings.append(np.asarray(coul) + np.asarray(mmp))

            # Before saving we must recover couplings below the threshold (they will be left as vac)
            all_couplings = np.array(predicted_couplings.copy())
            for i, env_coup in enumerate(threshold_mask):
                all_couplings[env_coup] = np.array(tot_couplings)[i]
            all_couplings = list(all_couplings)
            print_predicted_couplings(all_couplings, pairs_ids, kind="V_tot")
            save_coulomb_couplings(
                all_couplings, pairs_ids, kind="env_pol", outfile=args.outfile
            )


def _compute_vac_site_energies(molecules, args):
    """
    Compute the site energies in vacuum along a trajectory.

    Parameters
    ----------
    molecules: list of Molecule
        molecule instances
    args: argparse.Namespace
        CLI arguments

    Returns
    -------
    sites_vac: dict
        site energies in vacuum
        dictionary with keys "y_mean" and "y_var", each of which
        is a list of ndarrays (one for each chromophore).
    """
    sites_vac = defaultdict(list)

    for mol in molecules:
        site_vac = mol.vac_site_energy
        sites_vac["y_mean"].append(site_vac.value)
        sites_vac["y_var"].append(site_vac.var)
        # Save the site energy
        save_site_energies(
            site_vac.value,
            site_vac.var,
            mol.resid,
            kind="vac",
            outfile=args.outfile,
        )
    return sites_vac


def _compute_env_site_shift(molecules, args):
    """
    Parameters
    ----------
    molecules: list of Molecule
        molecule instances
    args: argparse.Namespace
        CLI arguments

    Returns
    -------
    site_env_shifts: dict
        electrochromic shift of the site energy
        dictionary with keys "y_mean" and "y_var", each of which
        is a list of ndarrays (one for each chromophore).
    """
    sites_env_shift = defaultdict(list)

    for mol in molecules:
        site_env_shift = mol.env_shift_site_energy
        sites_env_shift["y_mean"].append(site_env_shift.value)
        sites_env_shift["y_var"].append(site_env_shift.var)
        save_site_energies(
            site_env_shift.value,
            site_env_shift.var,
            mol.resid,
            kind="env_shift",
            outfile=args.outfile,
        )
    return sites_env_shift


def _compute_env_site_energies(molecules, args):
    """
    Parameters
    ----------
    molecules: list of Molecule
        molecule instances
    args: argparse.Namespace
        CLI arguments

    Returns
    -------
    site_env: dict
        site energy in environment
        dictionary with keys "y_mean" and "y_var", each of which
        is a list of ndarrays (one for each chromophore).
    """
    sites_env = defaultdict(list)
    for mol in molecules:
        site_env = mol.env_site_energy
        sites_env["y_mean"].append(site_env.value)
        sites_env["y_var"].append(site_env.var)
        save_site_energies(
            site_env.value,
            site_env.var,
            mol.resid,
            kind="env",
            outfile=args.outfile,
        )
    if (len(sites_env["y_mean"]) != len(args.residue_ids)) or (
        len(sites_env["y_var"]) != len(args.residue_ids)
    ):
        raise RuntimeError
    return sites_env


def _compute_mmp_site_shift(molecules, args):
    """
    Parameters
    ----------
    molecules: list of Molecule
        molecule instances
    args: argparse.Namespace
        CLI arguments

    Returns
    -------
    mmp_site_shifts: dict
        linear response contribution of the site energy in a polarizable env.
        dictionary with keys "y_mean" and "y_var", each of which
        is a list of ndarrays (one for each chromophore).
    """
    mmp_site_shifts = defaultdict(list)
    for mol in molecules:
        site = mol.pol_LR_site_energy
        mmp_site_shifts["y_mean"].append(site.value)
        mmp_site_shifts["y_var"].append(site.var)
        save_site_energies(
            site.value,
            site.var,
            mol.resid,
            kind="env_pol_shift",
            outfile=args.outfile,
        )
    if (len(mmp_site_shifts["y_mean"]) != len(args.residue_ids)) or (
        len(mmp_site_shifts["y_var"]) != len(args.residue_ids)
    ):
        raise RuntimeError
    return mmp_site_shifts


def _compute_mmp_site_energies(molecules, args):
    """
    Parameters
    ----------
    molecules: list of Molecule
        molecule instances
    args: argparse.Namespace
        CLI arguments

    Returns
    -------
    mmp_site_shifts: dict
        site energies in a polarizable environment
        dictionary with keys "y_mean" and "y_var", each of which
        is a list of ndarrays (one for each chromophore).
    """
    sites_mmp = defaultdict(list)
    for mol in molecules:
        site = mol.pol_site_energy
        sites_mmp["y_mean"].append(site.value)
        sites_mmp["y_var"].append(site.var)
        save_site_energies(
            site.value,
            site.var,
            mol.resid,
            kind="env_pol",
            outfile=args.outfile,
        )
    if (len(sites_mmp["y_mean"]) != len(args.residue_ids)) or (
        len(sites_mmp["y_var"]) != len(args.residue_ids)
    ):
        raise RuntimeError
    return sites_mmp


def compute_site_energies(molecules, args):
    """
    Predict site energies along a trajectory
    Arguments
    ---------
    molecules: list of Molecule
        molecule instances
    args: argparse.Namespace
        CLI arguments.
    """
    print_section("SITE ENERGIES @VAC")
    print_action("Encoding Geometries and Predicting Sites")
    # Encodings: Coulomb Matrices without permutations of atoms, hydrogen excluded
    # sites_vac: Site energies in vacuum.
    sites_vac = _compute_vac_site_energies(molecules, args)
    print_predicted_sitens(sites_vac["y_mean"], args.residue_ids, kind="E_vac")

    if not args.no_site_env:
        print_section("SITE ENERGIES @ENV")
        # Also here we use Coulomb Matrices with no hydrogens
        # for the internal part.
        # MM Potentials
        print_action(
            "Computing MM Electrostatic Potentials and Predicting Electrochromic Shift"
        )
        sites_env_shift = _compute_env_site_shift(molecules, args)
        print_predicted_sitens(
            sites_env_shift["y_mean"], args.residue_ids, kind="E_env_shift"
        )
        sites_env = _compute_env_site_energies(molecules, args)
        print_predicted_sitens(
            sites_env["y_mean"],
            args.residue_ids,
            kind="E_env",
        )

        if excipy.available_polarizable_module and not args.no_site_pol:
            print_section("SITE ENERGIES @MMPol")
            sites_mmp_shift = _compute_mmp_site_shift(molecules, args)
            print_predicted_sitens(
                sites_mmp_shift["y_mean"],
                args.residue_ids,
                kind="E_pol_shift",
            )
            sites_mmp = _compute_mmp_site_energies(molecules, args)
            print_predicted_sitens(
                sites_mmp["y_mean"],
                args.residue_ids,
                kind="E_pol",
            )


# =============================================================================
# Tools: Excipy2Exat
# =============================================================================


def excipy2exat_parse(argv):
    "Command line parser"
    parser = ArgumentParser(
        prog=sys.argv[0],
        description="Convert excipy output to EXAT formatted npz files.",
    )

    inpt = parser.add_argument_group("input")
    opt = inpt.add_argument

    opt(
        "-i",
        "--input",
        required=True,
        help="Excipy output (HDF5) file.",
    )

    opt(
        "-f",
        "--frames",
        required=False,
        nargs="+",
        default=[None, None, None],
        help="Frames to load, [start, stop, step] format, one-based count.",
    )

    opt(
        "-w",
        "--window",
        required=False,
        default=None,
        type=int,
        help="Window size (in frames) over which excitonic parameters are block-averaged.",
    )

    output = parser.add_argument_group("output")
    opt = output.add_argument

    opt(
        "-o",
        "--outfile",
        required=False,
        default="exat",
        help="Name of the output file(s) (without extension).",
    )

    units = parser.add_argument_group("units/kind")
    opt = units.add_argument

    opt(
        "--site_kind",
        required=False,
        default="vac",
        choices=[
            "vac",
            "env",
            "env_pol",
        ],
        help="Kind of site energy to collect (vacuum or environment)",
    )

    opt(
        "--coup_kind",
        required=False,
        default="vac",
        choices=[
            "vac",
            "env",
            "env_pol",
        ],
        help="Kind of coupling to collect (vacuum or environment)",
    )

    opt(
        "--dipo_kind",
        required=False,
        default="vac",
        choices=[
            "vac",
            "env",
        ],
        help="Kind of transition dipole to collect (vacuum or enviroment)",
    )

    opt(
        "--ene_units",
        required=False,
        default="cm_1",
        choices=[
            "cm_1",
            "eV",
        ],
        help="Units of energy (cm^-1 or eV)",
    )

    args = parser.parse_args(argv[1:])
    return args, parser


def _validate_window(window, n_frames):
    if window <= 1:
        raise ValueError("Block-averaging window cannot be negative or equal to 1.")
    if window > n_frames:
        raise RuntimeError(
            "Block-averaging window greater than total number of frames."
        )


def excipy2exat_block_average(exat_quantities, frames, args):
    print_action("Requested Block-Averaging of Excitonic Parameters.")
    print(": Validating window")
    _validate_window(args.window, len(frames))
    n_blocks = int(np.floor(len(frames) / args.window))
    rem = int(len(frames) % n_blocks)
    block_size = len(frames) // n_blocks
    print(f": Block-averaging over {n_blocks} blocks.")
    print(f": Last {rem} frames will no be considered in the average.")
    # Perform the block-averaging
    to_avg = ["site", "coup", "DipoLen", "DipoVel", "Mag"]
    for key in exat_quantities.keys():
        if key in to_avg:
            if exat_quantities[key] is None:
                print(f": {key} is stored as None: skipping.")
                pass
            else:
                print(f': Block-averaging "{key}"')
                exat_quantities[key] = block_average(
                    [np.asarray(exat_quantities[key])], n_blocks
                )[0]
        else:
            print(f': Skipping "{key}"')
    print(
        ": Extracting geometries (xyz) and centers (Cent) from the middle of each block."
    )
    # Coordinates get a separate treatment: we return the one in the middle
    # of the block, to have a meaningful geometry
    if rem != 0:
        exat_quantities["xyz"] = exat_quantities["xyz"][:-rem]
        exat_quantities["Cent"] = exat_quantities["Cent"][:-rem]
    exat_quantities["xyz"] = [
        a[int(block_size / 2)]
        for a in np.split(np.asarray(exat_quantities["xyz"]), n_blocks)
    ]
    exat_quantities["Cent"] = [
        a[int(block_size / 2)]
        for a in np.split(np.asarray(exat_quantities["Cent"]), n_blocks)
    ]
    return exat_quantities


def excipy2exat():
    logo()
    version()

    args, parser = excipy2exat_parse(sys.argv)
    print_cli_arguments(args)

    n_frames = load_n_frames(args.input)
    frames = parse_frames(args.frames, n_frames)

    print()

    exat_quantities = load_as_exat(
        args.input,
        frames=frames,
        sites_kind=args.site_kind,
        coups_kind=args.coup_kind,
        dipoles_kind=args.dipo_kind,
        ene_units=args.ene_units,
    )

    if args.window is not None:
        exat_quantities = excipy2exat_block_average(exat_quantities, frames, args)

    print_action("Dumping EXAT Files")
    dump_exat_files(exat_quantities, outname=args.outfile)

    print()


# =============================================================================
# Tools: Excipy-Scan
# =============================================================================


def excipy_scan_parse(argv):
    "Command line parser"
    parser = ArgumentParser(
        prog=sys.argv[0],
        description="Perform a scan, turning off residues one at the time, and computing the relative site energies in environment",
    )

    inpt = parser.add_argument_group("required")
    opt = inpt.add_argument

    opt(
        "-c",
        "--coordinates",
        required=True,
        help="Molecular Dynamics trajectory coordinates (.nc, .xtc, ...)",
    )

    opt(
        "-p",
        "--parameters",
        required=True,
        help="Molecular Dynamics parameters/topology (.prmtop, .gro, ...)",
    )

    opt(
        "-r",
        "--residue_id",
        required=True,
        help="Residue IDs (e.g., 1 to select the first residue)",
    )

    opt(
        "-t",
        "--turnoff_masks",
        required=True,
        nargs="+",
        help="Amber masks to turn off (no electrostatics) some part of the MM region (e.g., ':1' to turn off electrostatics of resid 1)",
    )

    optional = parser.add_argument_group("optional")
    opt = optional.add_argument

    opt(
        "-f",
        "--frames",
        required=False,
        nargs="+",
        default=[None, None, None],
        help="Frames to load, [start, stop, step] format, one-based count.",
    )

    opt(
        "--pol_cutoff",
        required=False,
        default=15.0,
        type=float,
        help="Polarization cutoff in Angstrom (no polarization for molecules farther than cutoff).",
    )

    opt(
        "--elec_cutoff",
        required=False,
        default=30.0,
        type=float,
        help="Cutoff for electrostatics in Angstrom.",
    )

    opt(
        "--no_site_pol",
        required=False,
        action="store_true",
        help="Whether to not compute the MMPol contribution to the site energies.",
    )

    opt(
        "--models",
        required=False,
        metavar="KEY=VALUE",
        nargs="+",
        help="Specify the desired model for each molecule.",
    )

    opt(
        "--database_folder",
        required=False,
        default=None,
        help="Absolute path to a custom database folder.",
    )

    opt(
        "--charges_db",
        required=False,
        default=None,
        help="database file containing static and pol charges and polarizabilities",
    )

    outfiles = parser.add_argument_group("output files")
    opt = outfiles.add_argument

    opt(
        "--outfile",
        required=False,
        default="scan.h5",
        help="Output file (HDF5).",
    )

    args = parser.parse_args(argv[1:])
    return args, parser


def excipy_scan():
    logo()
    version()

    args, parser = excipy_scan_parse(sys.argv)
    print_cli_arguments(args)

    models = parse_models(args.models)

    print_begin()

    if args.database_folder is not None:
        set_database_folder(args.database_folder)

    create_hdf5_outfile(args.outfile)
    save_residue_ids([args.residue_id], args.outfile)

    n_frames = pt.iterload(args.coordinates, top=args.parameters).n_frames
    frame_slice = parse_frames(args.frames, n_frames, return_slice=True)
    traj = pt.iterload(args.coordinates, top=args.parameters, frame_slice=frame_slice)
    save_n_frames(traj.n_frames, args.outfile)

    read_alphas = False if args.no_site_pol else True

    mol = Molecule(
        traj=traj,
        resid=args.residue_id,
        model_dict=models,
        elec_cutoff=args.elec_cutoff,
        pol_cutoff=args.pol_cutoff,
        turnoff_mask=None,
        charges_db=args.charges_db,
        template_mol2=None,
        read_alphas=read_alphas,
    )

    print_section("computing the vacuum site energy")

    site_vac = mol.vac_site_energy
    save_site_energies(
        site_vac.value, site_vac.var, mol.resid, kind="vac", outfile=args.outfile
    )

    # Note: loop over the turnoff masks and compute the site
    #   we could take advantage of the linearity of the potential
    #   and avoid recomputing it each time, but if we do so we
    #   should also implement a guard against residues that are
    #   outside the pol cutoff. keeping it simple here, but leaving
    #   this comment so you know how to make the code faster.
    for toff in args.turnoff_masks:

        print_section(
            "computing environment site energies for" f' turnoff mask "{toff:s}"'
        )
        # set base attribute, trigger recalc of cached properties
        mol.turnoff_mask = toff
        out_mask = mol.resid + "_toff_" + mol.turnoff_mask

        print_action("computing the environment site energy")
        # electrochromic shift
        site_env_shift = mol.env_shift_site_energy
        save_site_energies(
            site_env_shift.value,
            site_env_shift.var,
            out_mask,
            kind="env_shift",
            outfile=args.outfile,
        )

        # energy in environment
        site_env = mol.env_site_energy
        save_site_energies(
            site_env.value, site_env.var, out_mask, kind="env", outfile=args.outfile
        )

        if excipy.available_polarizable_module and not args.no_site_pol:

            print_action("computing the linear-response contribution")
            # linear response contribution
            site_pol_shift = mol.pol_LR_site_energy
            save_site_energies(
                site_pol_shift.value,
                site_pol_shift.var,
                out_mask,
                kind="pol_shift",
                outfile=args.outfile,
            )

            # energy in polarizable environment
            site_pol = mol.pol_site_energy
            save_site_energies(
                site_pol.value, site_pol.var, out_mask, kind="pol", outfile=args.outfile
            )


# =============================================================================
# Main
# =============================================================================


def main():
    logo()
    version()

    args, parser = cli_parse(sys.argv)
    print_cli_arguments(args)

    models = parse_models(args.models)

    print_begin()

    # If a custom database is specified, set the new database paths
    if args.database_folder is not None:
        set_database_folder(args.database_folder)

    create_hdf5_outfile(args.outfile)
    save_residue_ids(args.residue_ids, args.outfile)

    n_frames = pt.iterload(args.coordinates, top=args.parameters).n_frames
    frame_slice = parse_frames(args.frames, n_frames, return_slice=True)
    traj = pt.iterload(args.coordinates, top=args.parameters, frame_slice=frame_slice)
    save_n_frames(traj.n_frames, args.outfile)

    # Decide whether to read the polarizabilities

    if args.no_coup:
        read_alphas = not args.no_site_pol
    else:
        read_alphas = not (args.no_site_pol and args.no_coup_pol)

    # Define molecules
    molecules = [
        Molecule(
            traj=traj,
            resid=resid,
            model_dict=models,
            elec_cutoff=args.elec_cutoff,
            pol_cutoff=args.pol_cutoff,
            turnoff_mask=args.turnoff_mask,
            charges_db=args.charges_db,
            template_mol2=None,
            read_alphas=read_alphas,
        )
        for resid in args.residue_ids
    ]  # TODO: add possibility to provide charges_db and template_mol2

    # Print info on loaded molecules
    n_atoms = [mol.n_atoms for mol in molecules]
    types = [mol.type for mol in molecules]
    print_molecules(types, args.residue_ids, n_atoms)

    # Dump molecule coordinates
    print_action("Loading chromophore coordinates")
    coords = [mol.coords for mol in molecules]
    atnums = [mol.atnums for mol in molecules]

    # Dump coords to file
    save_coords(coords, args.residue_ids, args.outfile)
    save_atom_numbers(atnums, args.residue_ids, args.outfile)

    if not args.no_coup:
        compute_couplings(molecules, args)

    if not args.no_siten:
        compute_site_energies(molecules, args)

    print_end()


if __name__ == "__main__":
    main()
