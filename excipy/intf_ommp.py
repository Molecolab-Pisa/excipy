from __future__ import annotations
from typing import Tuple, Any
import warnings

import pytraj as pt

import numpy as np
from scipy.spatial.distance import cdist
import pyopenmmpol as ommp

from .clib.retain_full_residues import retain_full_residues_cy
from .polar import compute_polarizabilities
from .util import build_connectivity_matrix


def get_residues_array(topology, mm_indices):
    num_mm = np.array([len(mm_indices)])
    _, indices = np.unique(
        np.array([topology.atom(i).resid for i in mm_indices]), return_index=True
    )
    # This array is passed to C code: ensure it is an array of C integers
    residues_array = np.concatenate([indices, num_mm]).astype(np.intc)
    return residues_array


def cutoff_with_whole_residues(
    mm_topology: Any, qm_coords: np.ndarray, mm_coords: np.ndarray, cutoff: float
):
    """applies the cutoff without cutting the residues

    Applies a given cutoff around the QM atoms so that each residue is
    taken as a whole.

    Args:
        mm_topology: pytraj topology of just the mm part
        qm_coords: coordinates of the qm part
        mm_coords: coordinates of the mm part
        cutoff: cutoff for the mm part
    """
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
    # amber mask counts from 1
    mm_mask = "@" + ",".join((mm_idx + 1).astype(str))
    mm_top = mm_topology[mm_mask]
    return num_mm, mm_coords, mm_top, mm_idx


def read_electrostatics_from_top(top: str):
    """reads the electrostatic quantities from a pytraj topology

    Reads the static charges, the polarizable charges, and
    the polarizabilities from a pytraj topology using WangAL
    parameters for the polarizability.

    Note: polarizable charges are equal to the static charges
          as there's only one set of charges in the topology.

    Note: terminal groups may not be recognized.

    Args:
        top: pytraj topology
    """
    charges = np.array([atom.charge for atom in top.atoms])
    pol_charges = charges.copy()
    alphas = compute_polarizabilities(top)
    return charges, pol_charges, alphas


def read_electrostatics_from_db(
    top: Any, db: str
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
    def warn(resname, atomname, altatomname):
        warnings.warn(
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


def _filter_indices(*arrays, indices):
    return [arr[indices] for arr in arrays]


def write_mmp_input(
    fname: str,
    atomic_numbers: np.ndarray,
    coordinates: np.ndarray,
    resid: np.ndarray,
    static_charges: np.ndarray,
    polarizabilities: np.ndarray,
    connectivity: np.ndarray,
) -> None:
    """Writes the .mmp file

    Args:
        fname: file name
        atomic_numbers: atomic numbers, shape (num_atoms,)
        coordinates: coordinates, shape (num_atoms, 3)
        resid: residues, shape (num_atoms,)
        static_charges: static charges, shape (num_atoms,)
        polarizabilities: polarizabilities, shape (num_atoms,)
        connectivity: connectivity matrix, shape (num_atoms, 8)
    """
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
        """
        Args:
            mm_cut: cutoff for the MM part
            pol_cut: cutoff for the polarizable part
            top: pytraj Topology of the whole system
            qm_mask: Amber mask for the QM part
            mmp_fname: file name where the input mmp is written
            db: database file (qmip-like)
            mol2: template mol2 file (qmip-like)
        """
        self.mm_cut = mm_cut
        self.pol_cut = pol_cut
        self.top = top
        self.qm_mask = qm_mask
        self.mmp_fname = mmp_fname
        self.db = db
        self.mol2 = mol2

        # read electrostatics
        if db is None:
            warnings.warn(
                "Database file not provided. Pol charges will be the same as the charges"
                " found in the prmtop file. Maybe they are not compatible with polarizabilities."
                " Using WangAL polarizabilities."
            )
            self.charges, self.pol_charges, self.alphas = read_electrostatics_from_top(
                top=self.top
            )
        else:
            topology = self._maybe_read_top_from_mol2(mol2=self.mol2)
            charges, pol_charges, alphas = read_electrostatics_from_db(
                top=topology, db=self.db
            )
            self.charges = charges
            self.pol_charges = pol_charges
            self.alphas = alphas

        # read additional info
        self.atomic_numbers = self._get_atomic_numbers(top=self.top)
        self.resids = self._get_resids(top=self.top)

    def _maybe_read_top_from_mol2(self, mol2: str):
        if mol2 is None:
            warnings.warn(
                "Reading from database but mol2 not provided."
                " I may not be able to recognize terminal residues."
            )
            return self.top
        else:
            return pt.load_topology(mol2)

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

    def _get_atomic_numbers(self, top: Any):
        return np.array([atom.atomic_number for atom in top.atoms], copy=True)

    def _get_resids(self, top: Any):
        return np.array([atom.resid for atom in top.atoms], copy=True)

    def _get_selection_and_complement(self, top: Any, mask: str):
        compl_mask = "!" + mask
        idx = top.select(mask)
        compl_idx = top.select(compl_mask)
        return idx, compl_idx

    def potential_approx_ipd(self, coords: np.ndarray):
        """potential of approximate induced dipoles on the qm atoms

        Computes the potential of approximate induced dipoles on the qm
        atoms. The induced dipoles are termed "approximate" as the qm
        part is treated as a classical collection of static charges with
        no polarizability.

        Args:
            coords : coordinates of the whole system, shape (n_atoms, 3)

        Returns:
            elecpot: potential of the induced dipoles on the qm atoms,
                     shape (n_qm,)
            system: OMMPSystem object with the approximate induced dipoles
        """
        num_mm, qm_coords, mm_coords, mm_top, mm_idx = self._mm_cutoff(coords=coords)
        num_pol, pol_coords, pol_top, pol_idx = self._pol_cutoff(
            qm_coords=qm_coords, mm_coords=mm_coords, mm_top=mm_top
        )

        # indices of the qm part and the environment part in the original topology
        qm_idx, env_idx = self._get_selection_and_complement(
            top=self.top, mask=self.qm_mask
        )
        # indices of the polarizable atoms in the original topology
        pol_idx = env_idx[mm_idx][pol_idx]
        # retain the qm part and all the mm part
        retain_idx = np.sort(np.concatenate((qm_idx, env_idx[mm_idx])))

        # static charges
        static_charges = self.charges.copy()
        static_charges[pol_idx] = self.pol_charges[pol_idx].copy()

        # polarizabilities
        alpha = np.zeros(self.top.n_atoms, dtype=np.float64)
        alpha[pol_idx] = self.alphas[pol_idx].copy()

        # restrict the selection
        atomic_numbers, resids, coordinates, static_charges, alpha = _filter_indices(
            self.atomic_numbers,
            self.resids,
            coords,
            static_charges,
            alpha,
            indices=retain_idx,
        )

        #  topology with the qm and the mm parts
        qmmm_mask = (
            "(" + self.qm_mask + ") | @" + ",".join((env_idx[mm_idx] + 1).astype(str))
        )
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

        # zero field from the qm part because the qm is treated as mm
        system = ommp.OMMPSystem(self.mmp_fname)
        zero_ext_field = np.zeros((system.pol_atoms, 3))
        system.set_external_field(zero_ext_field)
        elecpot = system.potential_pol2ext(qm_coords)

        return elecpot, system


#     def approximate_induced_dipoles2(self, coords: np.ndarray):
#         "approximates the induced dipoles using static charges for the QM part"
#         num_mm, qm_coords, mm_coords, mm_top, mm_idx = self._mm_cutoff(coords=coords)
#         num_pol, pol_coords, pol_top, pol_idx = self._pol_cutoff(
#             qm_coords=qm_coords, mm_coords=mm_coords, mm_top=mm_top
#         )
#
#         # indices to select the environment from the topology
#         env_mask = "!" + self.qm_mask
#         env_idx = self.top.select(env_mask)
#         qm_idx = self.top.select(self.qm_mask)
#
#         # MM part
#         # get the atomic numbers
#         atomic_numbers = self._get_atomic_numbers(top=mm_top)
#         # coordinates are in mm_coords
#         # get the resids
#         resids = self._get_resids(top=mm_top)
#         # get the static charges
#         static_charges = self.charges[env_idx][mm_idx]
#         static_charges[pol_idx] = self.pol_charges[env_idx][mm_idx][pol_idx]
#         # compute the polarizabilities for the whole mm part, and set it to
#         # zero for the non-polarizable atoms
#         alpha = np.zeros((num_mm,))
#         alpha[pol_idx] = self.alphas[env_idx][mm_idx][pol_idx]
#         # alpha_pol = read_electrostatics_from_db(pol_top, self.db)[2]
#         # alpha[pol_idx] = alpha_pol
#         # hack: zero connectivity matrix
#         conn = build_connectivity_matrix(mm_top, count_as="python") + 1
#         maxc = conn.shape[1]
#         connectivity = np.zeros((num_mm, 8))
#         connectivity[:, :maxc] = conn
#
#         # # QM part
#         # qm_top = self.top[self.qm_mask]
#         # num_qm = qm_top.n_atoms
#         # # atomic numbers
#         # qm_atomic_numbers = self._get_atomic_numbers(top=qm_top)
#         # # coordinates are in qm_coords
#         # # resids (there may be an overlap with the mm ones since the
#         # # topology is cut, but should not cause problems)
#         # qm_resids = self._get_resids(top=qm_top)
#         # # static charges
#         # qm_idx = self.top.select(self.qm_mask)
#         # qm_static_charges = self.charges[qm_idx]
#         # # polarizabilities are zero
#         # qm_alpha = np.zeros((num_qm,))
#         # # hack: zero connectivity matrix
#         # qm_connectivity = np.zeros((num_qm, 8))
#
#         # # stick together
#         # atomic_numbers = np.concatenate((qm_atomic_numbers, atomic_numbers))
#         # coordinates = np.concatenate((qm_coords, mm_coords), axis=0)
#         # resids = np.concatenate((qm_resids, resids))
#         # static_charges = np.concatenate((qm_static_charges, static_charges))
#         # alpha = np.concatenate((qm_alpha, alpha))
#         # connectivity = np.concatenate((qm_connectivity, connectivity), axis=0)
#
#         coordinates = mm_coords  # removeme
#
#         write_mmp_input(
#             fname=self.mmp_fname,
#             atomic_numbers=atomic_numbers,
#             coordinates=coordinates,
#             resid=resids,
#             static_charges=static_charges,
#             polarizabilities=alpha,
#             connectivity=connectivity,
#         )
#
#         system = ommp.OMMPSystem(self.mmp_fname)
#         zero_ext_field = np.zeros((system.pol_atoms, 3))
#         system.set_external_field(zero_ext_field)
#         return system.ipd.copy(), system
