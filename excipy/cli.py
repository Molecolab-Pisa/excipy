#!/usr/bin/env python

import sys
import numpy as np
import pytraj as pt
from argparse import ArgumentParser
from collections import defaultdict

import excipy
from excipy.trajectory import parse_masks
from excipy.descriptors import (
    get_coulomb_matrix,
    get_MM_elec_potential,
    encode_geometry,
)
from excipy.regression import predict_tresp_charges, predict_site_energies
from excipy.database import (
    get_atom_names,
    get_hydrogens,
    get_identical_atoms,
    get_rescalings,
    select_masks,
    set_database_folder,
    get_site_model_params,
)
from excipy.elec import compute_coulomb_couplings
from excipy.util import (
    EV2CM,
    create_hdf5_outfile,
    save_n_frames,
    save_residue_ids,
    save_coords,
    save_atom_numbers,
    save_tresp_charges,
    save_dipoles,
    save_coulomb_couplings,
    save_site_energies,
    load_tresp_charges,
    load_n_frames,
    load_as_exat,
    dump_exat_files,
    rescale_tresp,
    read_molecule_types,
    Colors,
    get_dipoles,
    block_average,
)

if excipy.available_polarizable_module:
    from excipy.polar import batch_mmpol_site_lr, batch_mmpol_coup_lr


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


def cli_parse(argv):
    "Command line parser"
    parser = ArgumentParser(
        prog=sys.argv[0],
        description="Regression-based estimation of the Coulomb coupling.",
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
        "--database_folder",
        required=False,
        default=None,
        help="Absolute path to a custom database folder.",
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


def compute_couplings(traj, args):  # noqa: C901
    print_section("COULOMB COUPLINGS")

    # Get the residue types (residue names)
    topology = pt.load_topology(args.parameters)
    types = read_molecule_types(topology, args.residue_ids)
    # Load atom names from the database
    atom_names = get_atom_names(types)
    # Load groups of identical atoms from the database
    # (used for the coulomb matrix encoding for the couplings)
    permute_groups = get_identical_atoms(types)
    # Setup the selection masks
    masks = select_masks(args.residue_ids, atom_names)

    # Parse masks to obtain the coordinates
    print_action("Loading chromophore coordinates")
    coords, atnums = parse_masks(traj, masks, atom_names)

    # At this stage we can dump the coordinates in the output
    # as we are not omitting atoms
    save_coords(coords, args.residue_ids, args.outfile)
    save_atom_numbers(atnums, args.residue_ids, args.outfile)

    n_atoms = [c.shape[1] for c in coords]
    print_molecules(types, args.residue_ids, n_atoms)

    # Create a bunch of CoulombMatrix descriptors
    # These objects are needed as they store the permutation transformation
    coulomb_matrices = get_coulomb_matrix(
        coords=coords,
        atnums=atnums,
        residue_ids=args.residue_ids,
        permute_groups=permute_groups,
    )
    # Encode the molecular geometry as a Coulomb Matrix
    print_action("Encoding molecules")
    encodings = encode_geometry(coulomb_matrices, free_attr=["coords"])

    # Predict the TrEsp charges
    print_action("Predicting vacuum TrEsp charges")
    vac_tresp = predict_tresp_charges(
        encodings, coulomb_matrices, types, args.residue_ids
    )
    # Delete the encodings to save space as they are no more needed
    del encodings
    save_tresp_charges(vac_tresp, args.residue_ids, kind="vac", outfile=args.outfile)
    vac_tr_dipoles = get_dipoles(coords, vac_tresp)
    save_dipoles(vac_tr_dipoles, args.residue_ids, kind="vac", outfile=args.outfile)

    # Do not compute couplings if there is only one molecule
    if len(coords) < 2:
        pass
    else:
        print_action("Computing vacuum couplings")
        # Compute the couplings using the predicted TrEsp charges

        predicted_couplings, pairs_ids = compute_coulomb_couplings(
            coords, vac_tresp, args.residue_ids, args.cutoff, args.coup_list
        )

        print_predicted_couplings(predicted_couplings, pairs_ids, kind="V_vac")
        # No coupling to save if no molecule is within the cutoff distance
        if predicted_couplings is None:
            pass
        else:
            save_coulomb_couplings(
                predicted_couplings, pairs_ids, kind="vac", outfile=args.outfile
            )

    # We always provide an estimate of the TrEsp charges in environment
    print_action("Rescaling TrEsp and Coulomb couplings")
    env_scalings = get_rescalings(types)
    env_tresp = rescale_tresp(vac_tresp, env_scalings)
    save_tresp_charges(env_tresp, args.residue_ids, kind="env", outfile=args.outfile)
    env_tr_dipoles = get_dipoles(coords, env_tresp)
    save_dipoles(env_tr_dipoles, args.residue_ids, kind="env", outfile=args.outfile)

    # Do not compute couplings if there is only one molecule
    if len(coords) < 2:
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

            coup_scalings = [
                env_scalings[idx[0]] * env_scalings[idx[1]]
                for idx in above_threshold_pairs_idx
            ]

            env_couplings = np.asarray(
                [c * s for c, s in zip(above_threshold_couplings, coup_scalings)]
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
        if len(coords) < 2:
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
            mmpol_couplings, _ = batch_mmpol_coup_lr(
                traj,
                coords=coords,
                charges=env_tresp,
                residue_ids=args.residue_ids,
                masks=masks,
                pairs=above_threshold_pairs_idx,
                pol_threshold=args.pol_cutoff,
                db=None,
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


def _compute_vac_site_energies(coords, atnums, types, args):
    """
    Compute the site energies in vacuum along a trajectory.
    Arguments
    ---------
    coords    : list of ndarray, (num_frames, num_atoms, 3)
                Coordinates of the chromophores
    atnums    : list of ndarray, (num_atoms,)
                Atomic numbers of the chromophores
    types     : list of str
                Types of the chromophores
    args      : argparse.Namespace
                CLI arguments
    Returns
    -------
    encodings : list of ndarray, (num_frames, num_features)
                Encodings of the internal geometry
    sites_vac : dict or defaultdict
                Site energies in vacuum
                Dictionary with keys "y_mean" and "y_var", each of which
                is a list of ndarrays (one for each chromophore).
    """
    encodings = []
    sites_vac = defaultdict(list)

    # load the model parameters here to save time
    model_params = {t: get_site_model_params(t, kind="vac") for t in set(types)}

    for coord, atnum, residue_id, type in zip(coords, atnums, args.residue_ids, types):
        # Encode geometry returns a list: get the first (and only) element.
        # Same for `get_coulomb_matrix`.
        coul_mat = get_coulomb_matrix(
            coords=(coord,),
            atnums=(atnum,),
            residue_ids=(residue_id,),
            permute_groups=None,
        )
        encoding = encode_geometry(coul_mat)
        encodings.append(encoding[0])
        # Predict the site energy
        site_vac = predict_site_energies(
            encoding[0], type, model_params[type], residue_id, kind="vac"
        )
        # site_vac = predict_site_energies_old(encoding, (type,), (residue_id,), kind='vac')
        sites_vac["y_mean"].append(site_vac["y_mean"])
        sites_vac["y_var"].append(site_vac["y_var"])
        # Save the site energy
        save_site_energies(
            site_vac["y_mean"],
            site_vac["y_var"],
            residue_id,
            kind="vac",
            outfile=args.outfile,
        )
    return encodings, sites_vac


def _compute_env_site_shift(traj, masks, internal_encodings, types, args):
    """
    Arguments
    ---------
    traj       : pytraj.Trajectory
                 Trajectory object.
    masks      : list of str
                 Masks of the chromophores
    internal_encodings : list of ndarray, (num_frames, num_features)
                         Encodings of the internal geometry
    types      : list of str
                 Types of the chromophores
    args       : argparse.Namespace
                 CLI arguments
    Returns
    -------
    site_site_shifts : dict or defaultdict
                       MMPol contributions to the site energy
                       Dictionary with keys "y_mean" and "y_var", each of which
                       is a list of ndarrays (one for each chromophore).
    """
    sites_env_shift = defaultdict(list)

    # load the model parameters here to save time
    model_params = {t: get_site_model_params(t, kind="env") for t in set(types)}

    for internal_encoding, mask, type, residue_id in zip(
        internal_encodings, masks, types, args.residue_ids
    ):
        # Cutoff in Angstrom, hardcoded to 30. (our models are trained using this cutoff)
        # frames=None means all frames.
        # charges_db=None means take charges from topology
        elec_potential = get_MM_elec_potential(
            traj=traj,
            masks=(mask,),
            cutoff=30.0,
            frames=None,
            turnoff_mask=args.turnoff_mask,
            charges_db=None,
            remove_mean=False,
        )
        ep_encoding = encode_geometry(elec_potential)
        encoding = [
            np.column_stack([i, p]) for p, i in zip(ep_encoding, (internal_encoding,))
        ]
        site_env_shift = predict_site_energies(
            encoding[0], type, model_params[type], residue_id, kind="env"
        )
        sites_env_shift["y_mean"].append(site_env_shift["y_mean"])
        sites_env_shift["y_var"].append(site_env_shift["y_var"])
        save_site_energies(
            site_env_shift["y_mean"],
            site_env_shift["y_var"],
            residue_id,
            kind="env_shift",
            outfile=args.outfile,
        )
    return sites_env_shift


def _compute_env_site_energies(sites_vac, sites_env_shift, args):
    """
    Sum the @VAC, @ENV contributions to the site energies to obtain the total
    site energy in a non-polarizable environment.
    Arguments
    ---------
    sites_vac       : dict or defaultdict
                      Vacuum site energies.
                      Dictionary with keys "y_mean" and "y_var", each of which
                      is a list of ndarrays (one for each chromophore).
    sites_env_shift : dict or defaultdict
                      Environment site energies shifts
                      Dictionary with keys "y_mean" and "y_var", each of which
                      is a list of ndarrays (one for each chromophore).
    Returns
    -------
    sites_mmp       : dict or defaultdict
                      Site energies in a polarizable environment
                      Dictionary with keys "y_mean" and "y_var", each of which
                      is a list of ndarrays (one for each chromophore).
    """
    sites_env = dict()
    sites_env["y_mean"] = [
        vac_e + shift_e
        for vac_e, shift_e in zip(sites_vac["y_mean"], sites_env_shift["y_mean"])
    ]
    sites_env["y_var"] = [
        vac_e + shift_e
        for vac_e, shift_e in zip(sites_vac["y_var"], sites_env_shift["y_var"])
    ]
    if (len(sites_env["y_mean"]) != len(args.residue_ids)) or (
        len(sites_env["y_var"]) != len(args.residue_ids)
    ):
        raise RuntimeError
    for y_mean, y_var, residue_id in zip(
        sites_env["y_mean"], sites_env["y_var"], args.residue_ids
    ):
        save_site_energies(
            y_mean,
            y_var,
            residue_id,
            kind="env",
            outfile=args.outfile,
        )
    return sites_env


def _compute_mmp_site_shift(traj, coords, masks, types, args, charges):
    """
    Arguments
    ---------
    traj       : pytraj.Trajectory
                 Trajectory object.
    coords     : list of ndarray, (num_frames, num_atoms, 3)
                 Coordinates of the chromophores
    masks      : list of str
                 Masks of the chromophores
    types      : list of str
                 Types of the chromophores
    args       : argparse.Namespace
                 CLI arguments
    charges    : list of ndarray, (num_frames, num_atoms)
                 list of charges of the chromophores
    Returns
    -------
    mmp_site_shifts : dict or defaultdict
                      MMPol contributions to the site energy
                      Dictionary with keys "y_mean" and "y_var", each of which
                      is a list of ndarrays (one for each chromophore).
    """
    mmp_site_shifts, _ = batch_mmpol_site_lr(
        trajectory=traj,
        coords=coords,
        charges=charges,
        residue_ids=args.residue_ids,
        masks=masks,
        pol_threshold=args.pol_cutoff,
        db=None,
        mol2=None,
        cut_strategy="spherical",
        smear_link=True,
        turnoff_mask=args.turnoff_mask,
    )  # cm-1
    mmp_site_shifts = [(s / EV2CM).reshape(-1, 1) for s in mmp_site_shifts]
    mmp_site_shifts = {
        "y_mean": mmp_site_shifts,
        "y_var": [np.zeros(s.shape) for s in mmp_site_shifts],
    }
    if (len(mmp_site_shifts["y_mean"]) != len(args.residue_ids)) or (
        len(mmp_site_shifts["y_var"]) != len(args.residue_ids)
    ):
        raise RuntimeError
    for y_mean, y_var, residue_id in zip(
        mmp_site_shifts["y_mean"], mmp_site_shifts["y_var"], args.residue_ids
    ):
        save_site_energies(
            y_mean,
            y_var,
            residue_id,
            kind="env_pol_shift",
            outfile=args.outfile,
        )
    return mmp_site_shifts


def _compute_mmp_site_energies(sites_vac, sites_env_shift, sites_mmp_shift, args):
    """
    Sum the @VAC, @ENV, @POL contributions to the site energies to obtain the total
    site energy in a polarizable environment.
    Arguments
    ---------
    sites_vac       : dict or defaultdict
                      Vacuum site energies.
                      Dictionary with keys "y_mean" and "y_var", each of which
                      is a list of ndarrays (one for each chromophore).
    sites_env_shift : dict or defaultdict
                      Environment site energies shifts
                      Dictionary with keys "y_mean" and "y_var", each of which
                      is a list of ndarrays (one for each chromophore).
    sites_mmp_shift : dict or defaultdict
                      Polarizable site energies shifts
                      Dictionary with keys "y_mean" and "y_var", each of which
                      is a list of ndarrays (one for each chromophore).
    Returns
    -------
    sites_mmp       : dict or defaultdict
                      Site energies in a polarizable environment
                      Dictionary with keys "y_mean" and "y_var", each of which
                      is a list of ndarrays (one for each chromophore).
    """
    sites_mmp = dict()
    sites_mmp["y_mean"] = [
        vac_e + shift_e + shift_p
        for vac_e, shift_e, shift_p in zip(
            sites_vac["y_mean"], sites_env_shift["y_mean"], sites_mmp_shift["y_mean"]
        )
    ]
    sites_mmp["y_var"] = [
        vac_e + shift_e
        for vac_e, shift_e in zip(sites_vac["y_var"], sites_env_shift["y_var"])
    ]
    if (len(sites_mmp["y_mean"]) != len(args.residue_ids)) or (
        len(sites_mmp["y_var"]) != len(args.residue_ids)
    ):
        raise RuntimeError
    for y_mean, y_var, residue_id in zip(
        sites_mmp["y_mean"], sites_mmp["y_var"], args.residue_ids
    ):
        save_site_energies(
            y_mean,
            y_var,
            residue_id,
            kind="env_pol",
            outfile=args.outfile,
        )
    return sites_mmp


def compute_site_energies(traj, args):
    """
    Predict site energies along a trajectory
    Arguments
    ---------
    traj    : pytraj.Trajectory
              Trajectory object.
    args    : argparse.Namespace
              CLI arguments.
    """
    print_section("SITE ENERGIES @VAC")
    # Get the residue types (residue names)
    topology = pt.load_topology(args.parameters)
    types = read_molecule_types(topology, args.residue_ids)
    # Coulomb Matrix without hydrogens
    hydrogen_atoms = get_hydrogens(types)
    atom_names = get_atom_names(types, exclude_atoms=hydrogen_atoms)
    masks = select_masks(args.residue_ids, atom_names)
    print_action("Loading Coordinates")
    coords, atnums = parse_masks(traj, masks, atom_names)
    print_action("Encoding Geometries and Predicting Sites")
    # Encodings: Coulomb Matrices without permutations of atoms, hydrogen excluded
    # sites_vac: Site energies in vacuum.
    encodings, sites_vac = _compute_vac_site_energies(coords, atnums, types, args)
    print_predicted_sitens(sites_vac["y_mean"], args.residue_ids, kind="E_vac")

    if not args.no_site_env:
        print_section("SITE ENERGIES @ENV")
        # Also here we use Coulomb Matrices with no hydrogens
        # for the internal part.
        # MM Potentials
        atom_names = get_atom_names(types)
        masks = select_masks(args.residue_ids, atom_names)
        print_action(
            "Computing MM Electrostatic Potentials and Predicting Electrochromic Shift"
        )
        sites_env_shift = _compute_env_site_shift(traj, masks, encodings, types, args)
        print_predicted_sitens(
            sites_env_shift["y_mean"], args.residue_ids, kind="E_env_shift"
        )
        sites_env = _compute_env_site_energies(sites_vac, sites_env_shift, args)
        print_predicted_sitens(
            sites_env["y_mean"],
            args.residue_ids,
            kind="E_env",
        )

        if excipy.available_polarizable_module and not args.no_site_pol:
            print_section("SITE ENERGIES @MMPol")
            atom_names = get_atom_names(types)
            masks = select_masks(args.residue_ids, atom_names)
            coords, atnums = parse_masks(traj, masks, atom_names)
            # We try to obtain the environment TrEsp charges from the output file
            try:
                env_tresp = load_tresp_charges(args.outfile, kind="env")
            except KeyError:
                raise RuntimeError(
                    "Environment TrEsp not found. MMPol contribution to the "
                    " site energy is not computable."
                )

            charges = [env_tresp[r] for r in args.residue_ids]
            sites_mmp_shift = _compute_mmp_site_shift(
                traj=traj,
                coords=coords,
                masks=masks,
                types=types,
                args=args,
                charges=charges,
            )
            print_predicted_sitens(
                sites_mmp_shift["y_mean"],
                args.residue_ids,
                kind="E_pol_shift",
            )
            sites_mmp = _compute_mmp_site_energies(
                sites_vac, sites_env_shift, sites_mmp_shift, args
            )
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
# Main
# =============================================================================


def main():
    logo()

    args, parser = cli_parse(sys.argv)
    print_cli_arguments(args)

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

    if not args.no_coup:
        compute_couplings(traj, args)

    if not args.no_siten:
        compute_site_energies(traj, args)

    print_end()


if __name__ == "__main__":
    main()
