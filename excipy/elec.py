from __future__ import annotations
from typing import Dict, Tuple
import warnings

import numpy as np
import pytraj as pt

from .util import atomnum2atomsym, ANG2BOHR


# =============================================================================
# Polarization models
# for now only WangAL is supported.
# =============================================================================

WANGAL = {
    "B": 2.24204,  # Boron (J Phys Chem A, 103, 2141)
    "Mg": 0.1200,  # Magnesium (???)
    "C1": 1.3916,  # sp1 Carbon
    "C2": 1.2955,  # sp2 Carbon
    "C3": 0.9399,  # sp3 Carbon
    "H": 0.4255,  #
    "NO": 1.4824,  # Nitro Nitrogen
    "N": 0.9603,  # Other Nitrogen
    "O2": 0.6049,  # sp2 Oxygen
    "O3": 0.6148,  # sp3 Oxygen
    "F": 0.4839,  #
    "Cl": 2.3707,  #
    "Br": 3.5016,  #
    "I": 5.5788,  #
    "S4": 2.3149,  # S in sulfone
    "S": 3.1686,  # Other S
    "P": 1.7927,  #
    "Zn": 0.2600,  # Zinc (II) from AMOEBA (JCTC, 6, 2059)
}

WANGAL_FACTOR = 2.5874


# =============================================================================
# Utilities to compute polarizabilities
# =============================================================================


def _get_bonded_atoms(atom, topology):
    """
    Get the pytraj.Atom objects bonded to a pytraj.Atom.
    """
    return [topology.atom(i) for i in atom.bonded_indices()]


def _water_polarizability(atnum: int) -> float:
    """
    Get the atomic polarizability for a water atom.
    NOTE: currently not used but kept here for reference.
    """
    if atnum == 8:
        return 1.49073
    elif atnum == 1:
        return 0.0
    else:
        raise RuntimeError("Water atom is neither oxygen nor hydrogen.")


def _nitro_sulfone_symbol(topology: pt.Topology, atom: pt.Atom) -> str:
    """
    Get the atomic symbol for a nitro or sulfone atom
    """
    atnum = atom.atomic_number
    sym = atomnum2atomsym[atnum]
    bonded_atoms = _get_bonded_atoms(atom, topology)
    bonded_oxy_atoms = [a for a in bonded_atoms if a.atomic_number == 8]
    num_oxygens = len(bonded_oxy_atoms)
    if num_oxygens == 2:
        # We need two oxygens with valence one
        oxy_n_bonds = [a.n_bonds for a in bonded_oxy_atoms]
        if oxy_n_bonds == [1, 1]:
            # Real nitro/sulfone
            if sym == "N":
                return "NO"
            else:
                return "S4"
        else:
            return sym
    else:
        return sym


def compute_polarizabilities(
    topology: pt.Topology, poldict: Dict[str, float] = WANGAL
) -> np.ndarray:
    """
    Assign the atomic polarizability for each atom in the topology.
    Arguments
    ---------
    topology  : pytraj.Topology
              Trajectory topology
    poldict   : dict
              Dictionary mapping atom symbols to polarizabilities
    Returns
    -------
    polarizabilities : ndarray, (num_atoms,)
                     Atomic polarizabilities
    """
    atoms = topology.atoms
    polarizabilities = []
    for atom in atoms:
        atnum = atom.atomic_number
        # If uncommenting this if-else, the next "if"
        # resname = atom.resname
        # should become an "elif"
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        ## Waters get their own parameters                  # noqa: E266
        # if resname in ['WAT', 'WCR']:
        #    pol = _water_polarizability(atnum)
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Simple case, no ambiguities due to
        # hybridization of the atom:
        #            H  B  F  Mg  P   Cl  Br  I
        if atnum in [1, 5, 9, 12, 15, 17, 35, 53]:
            sym = atomnum2atomsym[atnum]
            pol = poldict[sym]
        # Cases with more than one type
        else:
            if atnum == 6:  # C
                sym = "C{:1d}".format(atom.n_bonds - 1)
            elif atnum == 8:  # O
                sym = "O{:1d}".format(atom.n_bonds + 1)
            # Distinguish nitro and sulfone
            elif atnum in [7, 16]:  # N, S
                sym = _nitro_sulfone_symbol(topology, atom)
            # Polarizability not present for this atom
            else:
                sym = None
            if sym is not None:
                pol = poldict[sym]
            else:
                pol = 0.0
        polarizabilities.append(pol)

    polarizabilities = np.array(polarizabilities)
    polarizabilities *= ANG2BOHR**3

    return polarizabilities


def read_electrostatics(
    top: pt.Topology, db: str = None, mol2: str = None, warn: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """reads the electrostatic parameters

    Reads the electrostatic parameters in multiple ways:

        (1) if a pytraj topology is passed:
            charges = topology charges
            pol_charges = charges
            alphas = WangAL polarizabilities
        (2) if a db is passed, reads these quantities from the db

    Note: if terminal residues do not have a different name in the topology,
          they cannot be identified. Use a mol2 together with a db for that case.

    Note: the mol2 is used only in conjunction with the db.

    Args:
        top: pytraj topology
        db: path to database file
        mol2: path to mol2 template file
        warn: whether to issue user warnings
    Returns:
        charges: atomic charges
        pol_charges: atomic charges compatible with polarizabilities
        alphas: polarizabilities
    """

    db_available = db is not None
    mol2_available = mol2 is not None

    if not db_available:
        if warn:
            warnings.warn(
                "Database file not provided. Pol charges will be the same as the"
                " charges found in the topology. Maybe they are not compatible with"
                " polarizabilities. Using WangAL polarizabilities."
            )
        charges, pol_charges, alphas = _read_electrostatics_from_top(top=top)

    elif db_available and not mol2_available:
        if warn:
            warnings.warn(
                "Database provided, but I don't have a mol2 template."
                " I may not be able to recognize terminal residues."
                " If this is a concern, please provide a mol2 template."
            )
        charges, pol_charges, alphas = _read_electrostatics_from_db(top=top, db=db)

    elif db_available and mol2_available:
        mol2_top = pt.load_topology(mol2)
        charges, pol_charges, alphas = _read_electrostatics_from_db(top=mol2_top, db=db)

    else:
        raise RuntimeError()

    return charges, pol_charges, alphas


def _read_electrostatics_from_top(
    top: pt.Topology,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """reads the electrostatic quantities from a pytraj topology

    Reads the static charges, the polarizable charges, and
    the polarizabilities from a pytraj topology using WangAL
    parameters for the polarizability.

    Note: polarizable charges are equal to the static charges
          as there's only one set of charges in the topology.

    Note: terminal groups may not be recognized. If this is a concern,
          use the database (db) interface instead.

    Args:
        top: pytraj topology
    """
    charges = np.array([atom.charge for atom in top.atoms])
    pol_charges = charges.copy()
    alphas = compute_polarizabilities(top)
    return charges, pol_charges, alphas


def _read_electrostatics_from_db(
    top: pt.Topology, db: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """reads the electrostatics quantities from a database file

    Reads the static charges, the polarizable charges, and the
    polarizabilities from a database file.

    Note: use a template mol2 file as a topology for compatibility
          with qmip.

    Args:
        top: topology
        db: path to database
    """
    charges = []
    pol_charges = []
    alphas = []

    db = np.loadtxt(db, usecols=[0, 1, 2, 3, 4], dtype=str)
    residues = db[:, 0].astype(str)
    atnames = db[:, 1].astype(str)
    db_charges = db[:, 2].astype(float)
    db_pol_charges = db[:, 3].astype(float)
    db_alphas = db[:, 4].astype(float)
    # Array of 'RESIDUE_NAME ATOM_NAME' elements
    resatname = np.array([res + " " + name for res, name in zip(residues, atnames)])

    for atom in top.atoms:
        try:
            res = top.residue(atom.resid).name
            name = atom.name
            pattern = res.strip() + " " + name.strip()
            idx = np.where(resatname == pattern)[0][0]
            charge = float(db_charges[idx])
            pol_charge = float(db_pol_charges[idx])
            alpha = float(db_alphas[idx])
        except IndexError:
            try:
                altname = _find_alternative_name(res, name)
                if altname is None:
                    msg = f"Charge for {res} {name} not found."
                    raise IndexError(msg)
                pattern = res.strip() + " " + altname.strip()
                idx = np.where(resatname == pattern)[0][0]
                charge = float(db_charges[idx])
                pol_charge = float(db_pol_charges[idx])
                alpha = float(db_alphas[idx])
            except IndexError:
                msg = f"Charge for {res} {name} not found."
                raise IndexError(msg)

        charges.append(charge)
        pol_charges.append(pol_charge)
        alphas.append(alpha)

    return np.array(charges), np.array(pol_charges), np.array(alphas)


def _find_alternative_name(resname, atomname):
    def warn(resname, atomname, altatomname):
        warnings.warn(
            f"Atom {atomname} for residue {resname} is treated as atom {altatomname}"
        )

    if atomname == "OXT":
        warn(resname, atomname, "O")
        return "O"


# =============================================================================
# Electrostatic quantities
# =============================================================================


def electric_field(
    target_coords: np.ndarray, source_coords: np.ndarray, source_charges: np.ndarray
) -> np.ndarray:
    """
    Computes the electric fields at points `target_coords`
    produced by a point charge distribution of point charges
    `source_charges` localized at `source_coords`.
    Arguments
    ---------
    target_coords  : ndarray, (num_targets, 3)
                   Target coordinates
    source_coords  : ndarray, (num_sources, 3)
                   Source coordinates
    source_charges : ndarray, (num_sources,)
                   Source charges
    Returns
    -------
    E              : ndarray, (num_targets, 3)
                   Electric field at the target coordinates
    """
    dist = target_coords[:, None, :] - source_coords[None, :, :]
    dist2 = np.sum(dist**2, axis=2)
    dist3 = dist2 * dist2**0.5
    ddist3 = dist / dist3[:, :, None]
    E = np.sum(ddist3 * source_charges[None, :, None], axis=1)
    return E


# def findaltnames(longresname, resname, aname):
#   # This is empirical!
#   alt = []
#   blk = False # Whether to remove "aname" from the DB
#   if resname in amino+nter_amino+['NME']:
#     if aname == 'HN': alt += ['H']
#   if resname == 'ILE' and aname == 'HD':
#      alt += ['HD1']
#   if resname == 'ARG' and aname == 'HH':
#      alt += ['HH1','HH2']
#   if resname in cter_amino:
#     if aname == 'OXT': alt += ['O']
#   if resname in ('ACE','NME') and aname == 'H':
#     alt += ['HH31','HH32','HH33']
#     blk  = True
#   if resname in ('Cl-', 'CL') and aname == 'CL':
#     alt += ['Cl-', 'CL-']
#   if resname in ('Na+', 'NA') and aname == 'NA':
#     alt += ['Na+', 'NA+']
#
#   # Paranoid check
#   if len(alt) == 0 and blk:
#     c.error('Could not find a valid atom name for %s %s' % (resname,aname) )
#   return alt,blk
