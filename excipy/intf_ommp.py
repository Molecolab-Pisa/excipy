from __future__ import annotations
from typing import Any, List
import os
import multiprocessing
from itertools import repeat

import numpy as np
import pyopenmmpol as ommp

from .util import build_connectivity_matrix, ANG2BOHR
from .selection import _mm_cutoff, _pol_cutoff
from .elec import read_electrostatics


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
        qm_charges: np.ndarray = None,
        db: str = None,
        mol2: str = None,
        ommp_set_verbose: int = 1,
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
        self.qm_charges = qm_charges
        self.db = db
        self.mol2 = mol2
        self.ommp_set_verbose = ommp_set_verbose

        ommp.set_verbose = ommp_set_verbose

        self.charges, self.pol_charges, self.alphas = read_electrostatics(
            top=self.top,
            db=self.db,
            mol2=self.mol2,
        )

        # read additional info
        self.atomic_numbers = self._get_atomic_numbers(top=self.top)
        self.resids = self._get_resids(top=self.top)

    def _get_atomic_numbers(self, top: Any):
        return np.array([atom.atomic_number for atom in top.atoms], copy=True)

    def _get_resids(self, top: Any):
        return np.array([atom.resid for atom in top.atoms], copy=True)

    def potential_approx_ipd(
        self,
        coords: np.ndarray,
        mmp_postfix: int = None,
        return_system: bool = False,
        pin_cpus: List[int] = None,
    ):
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
        static_charges = self.charges.copy()
        pol_charges = self.pol_charges.copy()
        alphas = self.alphas.copy()


        qm_coords, mm_coords, qm_idx, mm_idx, mm_top = _mm_cutoff(
            topology=self.top, coords=coords, qm_mask=self.qm_mask, mm_cut=self.mm_cut
        )

        pol_coords, pol_top, pol_idx = _pol_cutoff(
            mm_topology=mm_top,
            qm_coords=qm_coords,
            mm_coords=mm_coords,
            pol_cut=self.pol_cut,
        )
        # pol indices in the original topology
        pol_idx = mm_idx[pol_idx]

        # put the qm charges in the static charges if you have them
        if self.qm_charges is not None:
            static_charges[qm_idx] = self.qm_charges

        charge_qm = static_charges[qm_idx]

        # smear the charges for the eventual presence of link atoms
        charges, pol_charges, alphas = link_atom_smear(top=self.top, qm_idx=qm_idx, charges=static_charges, pol_charges=pol_charges, alphas=alphas)

        # static charges
        static_charges[pol_idx] = pol_charges[pol_idx].copy()

        # polarizabilities
        alpha = np.zeros(self.top.n_atoms, dtype=np.float64)
        alpha[pol_idx] = alphas[pol_idx].copy()

        # restrict the selection
        atomic_numbers, resids, coordinates, static_charges, alpha = _filter_indices(
            self.atomic_numbers,
            self.resids,
            coords,
            static_charges,
            alpha,
            indices=mm_idx,
        )

        conn = build_connectivity_matrix(mm_top, count_as="python") + 1
        maxc = conn.shape[1]
        connectivity = np.zeros((mm_top.n_atoms, 8))
        connectivity[:, :maxc] = conn

        if mmp_postfix is None:
            mmp_fname = self.mmp_fname
        else:
            mmp_fname, ext = os.path.splitext(self.mmp_fname)
            mmp_fname = mmp_fname + "_{:07d}".format(mmp_postfix) + ext

        write_mmp_input(
            fname=mmp_fname,
            atomic_numbers=atomic_numbers,
            coordinates=coordinates,
            resid=resids,
            static_charges=static_charges,
            polarizabilities=alpha,
            connectivity=connectivity,
        )

        system = ommp.OMMPSystem(mmp_fname)

        # the qm part is treated as mm (collection of point charges)
        # we compute the external field as the field of that collection of charges
        qm_helper = ommp.OMMPQmHelper(coord_qm=qm_coords*ANG2BOHR, charge_qm=charge_qm, z_qm=self.atomic_numbers[qm_idx])
        qm_helper.prepare_qm_ele_ene(system)
        system.set_external_field(qm_helper.E_n2p)

        # keep in atomic units to be consistent with the
        # electrostatic potential descriptor
        elecpot = system.potential_pol2ext(qm_coords * ANG2BOHR)  # [e / Bohr]

        # clean after yourself
        os.remove(mmp_fname)

        if return_system:
            return elecpot, system

        return elecpot

    def potential_approx_ipd_along_traj(
        self, traj: np.ndarray, parallel=False, nprocs=4
    ):
        if parallel:
            pool = multiprocessing.Pool(processes=nprocs)
            pot = pool.starmap(
                self.potential_approx_ipd,
                zip(traj, range(traj.shape[0]), repeat(False)),
            )
            pool.close()
            pool.join()
            pot = np.asarray(pot)
            return pot
        else:
            pot = []
            for coords in traj:
                pot_ = self.potential_approx_ipd(
                    coords, mmp_postfix=None, return_system=False
                )
                pot.append(pot_)
            pot = np.asarray(pot)
            return pot
