from __future__ import annotations
from typing import Tuple
import warnings

import pytraj as pt

import numpy as np
from scipy.spatial.distance import cdist
import pyopenmmpol as ommp

from .clib.retain_full_residues import retain_full_residues_cy
from .polar import (
    compute_polarizabilities,
    compute_environment_mask,
    _compute_polarizabilities,
)
from .util import build_connectivity_matrix


def get_residues_array(topology, mm_indices):
    num_mm = np.array([len(mm_indices)])
    _, indices = np.unique(
        np.array([topology.atom(i).resid for i in mm_indices]), return_index=True
    )
    # This array is passed to C code: ensure it is an array of C integers
    residues_array = np.concatenate([indices, num_mm]).astype(np.intc)
    return residues_array


def cutoff_with_whole_residues(mm_topology, qm_coords, mm_coords, cutoff):
    mm_idx = np.arange(mm_topology.n_atoms)
    residues_array = get_residues_array(mm_topology, mm_idx)
    # Compute the distances between each atom of the pigment
    # pair and every other atom of the environment
    # dist is of shape (num_env_atoms, num_pair_atoms)
    dist = cdist(mm_coords, qm_coords)
    # For each environment atom, find the minimum distance
    # from the pair coordinates
    # min_dist is of shape (num_env_atoms,)
    min_dist = np.min(dist, axis=1)
    # Keep atoms within the cutoff
    mm_mask = min_dist <= cutoff
    cut_mask = np.zeros(len(min_dist), dtype=bool)
    cut_mask[mm_mask] = True
    # mm mask now has the residues as a whole, within the cutoff
    mm_mask = retain_full_residues_cy(cut_mask.astype(np.intc), residues_array).astype(
        bool
    )
    num_mm = sum(mm_mask)
    mm_coords = mm_coords[mm_mask]
    # original indices of the selected mm atoms in the full mm topology
    mm_idx = mm_idx[mm_mask]
    # amber mask starts from 1
    mm_mask = "@" + ",".join((mm_idx + 1).astype(str))
    mm_top = mm_topology[mm_mask]
    return num_mm, mm_coords, mm_top, mm_idx


def read_electrostatics_from_db(
    top: Any, db: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
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
                altname = find_alternative_name(res, name)
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


def find_alternative_name(resname, atomname):
    warn = lambda resname, atomname, altatomname: warnings.warn(
        f"Atom {atomname} for residue {resname} is treated as atom {altatomname}"
    )
    if atomname == "OXT":
        warn(resname, atomname, "O")
        return "O"


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


def write_mmp_input(
    fname: str,
    atomic_numbers: np.ndarray,
    coordinates: np.ndarray,
    resid: np.ndarray,
    static_charges: np.ndarray,
    polarizabilities: np.ndarray,
    connectivity: np.ndarray,
) -> None:
    with open(fname, "w") as f:
        # mmp revision
        f.write("{:d}\n".format(2))
        # mmp job
        f.write("{:d}\n".format(0))
        # mmp verbosity
        f.write("{:d}\n".format(0))
        # ff type
        f.write("{:d}\n".format(0))
        # ff amber
        f.write("{:d}\n".format(0))
        # mmdamp
        f.write("{:d}\n".format(0))
        # damping radius
        f.write("{:16.8E}\n".format(0))
        # mmsolv
        f.write("{:d}\n".format(0))
        # mvec
        f.write("{:d}\n".format(0))
        # mm convergence
        f.write("{:d}\n".format(8))
        # fmm accuracy
        f.write("{:d}\n".format(0))
        # mmp-mmp fmm box
        f.write("{:16.8E}\n".format(12))
        # mmp-dd fmm box
        f.write("{:d}\n".format(0))
        # dd lmax
        f.write("{:d}\n".format(10))
        # dd ngrid
        f.write("{:d}\n".format(302))
        # convergence
        f.write("{:d}\n".format(10))
        # epsilon statical
        f.write("{:16.8E}\n".format(78.3553))
        # smoothing factor
        f.write("{:16.8E}\n".format(0.1))
        # cavity type
        f.write("{:d}\n".format(0))
        # probe radius
        f.write("{:16.8E}\n".format(1.4))
        # num mm atoms
        f.write("{:d}\n".format(coordinates.shape[0]))
        # boh
        f.write("{:d}\n".format(0))
        # atomic numbers
        np.savetxt(f, atomic_numbers, fmt="%d")
        # coordinates
        np.savetxt(f, coordinates, fmt="%12.6f")
        # resids
        np.savetxt(f, resid, fmt="%d")
        # charges
        np.savetxt(f, static_charges, fmt="%12.6f")
        # polarizabilities
        np.savetxt(f, polarizabilities, fmt="%12.6f")
        # connectivity matrix
        np.savetxt(f, connectivity, fmt="%d")


class OMMPInterface:

    def __init__(
        self,
        mm_cut: float,
        pol_cut: float,
        top: Any,
        qm_mask: str,
        mmp_fname: str,
        db: str = None,
        mol2: str = None,
    ) -> None:
        self.mm_cut = mm_cut
        self.pol_cut = pol_cut
        self.top = top
        self.qm_mask = qm_mask
        self.mmp_fname = mmp_fname
        self.db = db
        self.mol2 = mol2

        if db is None:
            warnings.warn(
                "Database file not provided. Pol charges will be the same as the charges"
                " found in the prmtop file. Maybe they are not compatible with polarizabilities."
                " Using WangAL polarizabilities."
            )
            self.charges = self._get_static_charges(top=self.top)
            self.pol_charges = self.charges.copy()
            self.alphas = compute_polarizabilities(self.top)
        else:
            if mol2 is None:
                warnings.warn(
                    "Reading from database but mol2 not provided. I may not be able to recognize terminal residues."
                )
                topology = self.top
            else:
                topology = pt.load_topology(mol2)

            charges, pol_charges, alphas = read_electrostatics_from_db(
                top=topology, db=self.db
            )
            self.charges = charges
            self.pol_charges = pol_charges
            self.alphas = alphas

    def _mm_cutoff(self, coords: np.ndarray):
        mm_mask = "!" + self.qm_mask
        mm_top = self.top[mm_mask]
        qm_coords = coords[self.top.select(self.qm_mask)]
        mm_coords = coords[self.top.select(mm_mask)]

        num_mm, mm_coords, mm_top, mm_idx = cutoff_with_whole_residues(
            mm_topology=mm_top,
            qm_coords=qm_coords,
            mm_coords=mm_coords,
            cutoff=self.mm_cut,
        )
        return num_mm, qm_coords, mm_coords, mm_top, mm_idx

    def _pol_cutoff(self, qm_coords: np.ndarray, mm_coords: np.ndarray, mm_top: Any):
        "Note: this must be done after a call to self._mm_cut to save time!"
        num_pol, pol_coords, pol_top, pol_idx = cutoff_with_whole_residues(
            mm_topology=mm_top,
            qm_coords=qm_coords,
            mm_coords=mm_coords,
            cutoff=self.pol_cut,
        )
        return num_pol, pol_coords, pol_top, pol_idx

    def _get_static_charges(self, top: Any):
        return np.array([atom.charge for atom in top.atoms], copy=True)

    def _get_atomic_numbers(self, top: Any):
        return np.array([atom.atomic_number for atom in top.atoms], copy=True)

    def _get_resids(self, top: Any):
        return np.array([atom.resid for atom in top.atoms], copy=True)

    def approximate_induced_dipoles(self, coords: np.ndarray):
        "approximates the induced dipoles using static charges for the QM part"
        num_mm, qm_coords, mm_coords, mm_top, mm_idx = self._mm_cutoff(coords=coords)
        num_pol, pol_coords, pol_top, pol_idx = self._pol_cutoff(
            qm_coords=qm_coords, mm_coords=mm_coords, mm_top=mm_top
        )

        # indices to select the environment from the topology
        env_mask = "!" + self.qm_mask
        env_idx = self.top.select(env_mask)
        qm_idx = self.top.select(self.qm_mask)

        atomic_numbers = self._get_atomic_numbers(top=self.top)
        resids = self._get_resids(top=self.top)
        # static charges
        static_charges = self.charges.copy()
        static_charges[env_idx[mm_idx][pol_idx]] = self.pol_charges[env_idx][mm_idx][
            pol_idx
        ].copy()
        # polarizabilities
        alpha = np.zeros(self.top.n_atoms, dtype=np.float64)
        alpha[env_idx[mm_idx][pol_idx]] = self.alphas[env_idx][mm_idx][pol_idx].copy()

        # trim the part of interest
        retain_idx = np.sort(np.concatenate((qm_idx, env_idx[mm_idx])))
        # retain_idx = np.sort(env_idx[mm_idx])

        atomic_numbers = atomic_numbers[retain_idx]
        resids = resids[retain_idx]
        coordinates = coords[retain_idx]
        static_charges = static_charges[retain_idx]
        alpha = alpha[retain_idx]

        # connectivity
        #  we take a topology with qm + mm
        qmmm_mask = (
            "(" + self.qm_mask + ") | @" + ",".join((env_idx[mm_idx] + 1).astype(str))
        )
        # qmmm_mask = '@' + ','.join((env_idx[mm_idx]+1).astype(str))

        qmmm_top = self.top[qmmm_mask]
        conn = build_connectivity_matrix(qmmm_top, count_as="python") + 1
        maxc = conn.shape[1]
        connectivity = np.zeros((qmmm_top.n_atoms, 8))
        connectivity[:, :maxc] = conn

        write_mmp_input(
            fname=self.mmp_fname,
            atomic_numbers=atomic_numbers,
            coordinates=coordinates,
            resid=resids,
            static_charges=static_charges,
            polarizabilities=alpha,
            connectivity=connectivity,
        )

        system = ommp.OMMPSystem(self.mmp_fname)
        zero_ext_field = np.zeros((system.pol_atoms, 3))
        system.set_external_field(zero_ext_field)
        return system.ipd.copy(), system

    def approximate_induced_dipoles2(self, coords: np.ndarray):
        "approximates the induced dipoles using static charges for the QM part"
        num_mm, qm_coords, mm_coords, mm_top, mm_idx = self._mm_cutoff(coords=coords)
        num_pol, pol_coords, pol_top, pol_idx = self._pol_cutoff(
            qm_coords=qm_coords, mm_coords=mm_coords, mm_top=mm_top
        )

        # indices to select the environment from the topology
        env_mask = "!" + self.qm_mask
        env_idx = self.top.select(env_mask)
        qm_idx = self.top.select(self.qm_mask)

        # MM part
        # get the atomic numbers
        atomic_numbers = self._get_atomic_numbers(top=mm_top)
        # coordinates are in mm_coords
        # get the resids
        resids = self._get_resids(top=mm_top)
        # get the static charges
        static_charges = self.charges[env_idx][mm_idx]
        static_charges[pol_idx] = self.pol_charges[env_idx][mm_idx][pol_idx]
        # compute the polarizabilities for the whole mm part, and set it to
        # zero for the non-polarizable atoms
        alpha = np.zeros((num_mm,))
        alpha[pol_idx] = self.alphas[env_idx][mm_idx][pol_idx]
        # alpha_pol = read_electrostatics_from_db(pol_top, self.db)[2]
        # alpha[pol_idx] = alpha_pol
        # hack: zero connectivity matrix
        conn = build_connectivity_matrix(mm_top, count_as="python") + 1
        maxc = conn.shape[1]
        connectivity = np.zeros((num_mm, 8))
        connectivity[:, :maxc] = conn

        # # QM part
        # qm_top = self.top[self.qm_mask]
        # num_qm = qm_top.n_atoms
        # # atomic numbers
        # qm_atomic_numbers = self._get_atomic_numbers(top=qm_top)
        # # coordinates are in qm_coords
        # # resids (there may be an overlap with the mm ones since the
        # # topology is cut, but should not cause problems)
        # qm_resids = self._get_resids(top=qm_top)
        # # static charges
        # qm_idx = self.top.select(self.qm_mask)
        # qm_static_charges = self.charges[qm_idx]
        # # polarizabilities are zero
        # qm_alpha = np.zeros((num_qm,))
        # # hack: zero connectivity matrix
        # qm_connectivity = np.zeros((num_qm, 8))

        # # stick together
        # atomic_numbers = np.concatenate((qm_atomic_numbers, atomic_numbers))
        # coordinates = np.concatenate((qm_coords, mm_coords), axis=0)
        # resids = np.concatenate((qm_resids, resids))
        # static_charges = np.concatenate((qm_static_charges, static_charges))
        # alpha = np.concatenate((qm_alpha, alpha))
        # connectivity = np.concatenate((qm_connectivity, connectivity), axis=0)

        coordinates = mm_coords  # removeme

        write_mmp_input(
            fname=self.mmp_fname,
            atomic_numbers=atomic_numbers,
            coordinates=coordinates,
            resid=resids,
            static_charges=static_charges,
            polarizabilities=alpha,
            connectivity=connectivity,
        )

        system = ommp.OMMPSystem(self.mmp_fname)
        zero_ext_field = np.zeros((system.pol_atoms, 3))
        system.set_external_field(zero_ext_field)
        return system.ipd.copy(), system
