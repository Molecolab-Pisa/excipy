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
import cython
cimport numpy as np
import numpy as np
from libc.math cimport sqrt, exp
from cython.parallel import prange

@cython.boundscheck(False)
@cython.wraparound(False)
cdef bint do_inter(int i, int j, np.int64_t[:, ::1] nn_list, int nat, int nmax) nogil:
    if (i==j):
        return False
    cdef int l
    for l in range(nmax):
        if (nn_list[i,l]==j+1):
            return False
    return True


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void update_efi(int nat, int nmax, int i, np.int64_t[:, ::1] nn_list,
                double[:,::1] crd, double[::1] thole, double[:,::1] mu, int iscreen, double *E) nogil:

    cdef int j, k
    cdef double rvec[3]
    cdef double r2, r, r3, r5, s, v, v3, lambda5, lambda3, rm3, rm5, scd, fexp

    for j in range(nat):

            if not do_inter(i, j, nn_list, nat, nmax):
                continue

            for k in range(3):
                rvec[k] = crd[i,k] - crd[j,k]

            r2 = rvec[0]*rvec[0] + rvec[1]*rvec[1] + rvec[2]*rvec[2]
            r = sqrt(r2)
            r3 = r * r2
            r5 = r2 * r3

            # no screening
            if iscreen == 0:
                lambda3 = 1.
                lambda5 = 1.
            # thole screening
            elif iscreen == 1:
                s = thole[i]*thole[j]
                if (r<s):
                    v = r/s
                    v3 = v*v*v
                    lambda5 = v3*v
                    lambda3 = 4.*v3 - 3.*lambda5
                else:
                    lambda3 = 1.
                    lambda5 = 1.
            # exponential thole screening
            elif iscreen == 2:
                s = thole[i]*thole[j]
                v = r/s
                fexp = - v*v*v
                if fexp >= -50.0:
                    ef = exp(fexp)
                    lambda3 = 1. - ef
                    lambda5 = 1. - (1. - fexp)*ef
                else:
                    lambda3 = 1.
                    lambda5 = 1.

            rm3 = lambda3/r3
            rm5 = 3.*lambda5/r5
            scd = mu[j,0]*rvec[0] + mu[j,1]*rvec[1] + mu[j,2]*rvec[2]

            for k in range(3):
                E[k] +=  mu[j,k]*rm3 - scd*rm5*rvec[k]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double[:, ::1] _tmu(
    double[:,::1] mu,
    double[::1] alpha,
    double[::1] thole,
    np.int64_t[:, ::1] nn_list,
    double[:, ::1] crd,
    int iscreen,
):
    cdef int nat = crd.shape[0]
    cdef int nmax = nn_list.shape[1]
    cdef int i

    _E = np.zeros((nat,3), dtype=np.float64)
    cdef double[:, ::1] E = _E

    for i in prange(nat, nogil=True):
        update_efi(nat, nmax, i, nn_list, crd, thole, mu, iscreen, &E[i,0])

    return E

@cython.embedsignature(True)
def tmu_cy(double[:,:] mu, double[:] alpha, double[:] thole, np.int64_t[:,:] nn_list, double[:,:] crd, int iscreen):
    """
    Computes E = T*mu, where T is the MMPol matrix (excluding the diagonal)
    and mu are input dipoles.

    Some elements of T are zeroed or screened according to the
    Thole smeared dipole formulation.

    Distance-dependent screening factors and factors for nearest neighbours
    are taken from the AL model of Wang et al. JPC B (2011), 115, 3091

    Arguments
    ---------
    mu: np.ndarray, (num_pol_atoms, 3)
        induced dipoles.
    alpha: np.ndarray, (num_pol_atoms,)
        polarizabilities.
    thole: np.ndarray, (num_pol_atoms,)
        thole parameters.
    nn_list: np.ndarray, (num_pol_atoms, nnmax)
        nearest-neighbor matrix. The i-th row stores the indices (1-based count
        for "historical" reasons) of the neighbors of atom i (0-based count on
        the rows).
    crd: np.ndarray, (num_pol_atoms, 3)
        coordinates of the polarizable atoms.
    iscreen: int
        which screening factor to use.
        - 0 for no screening
        - 1 for thole screening
        - 2 for exponential thole screening

    Returns
    -------
    E: np.ndarray, (num_pol_atoms, 3)
        electric field.
    """
    return _tmu(
        np.ascontiguousarray(mu),
        np.ascontiguousarray(alpha),
        np.ascontiguousarray(thole),
        np.ascontiguousarray(nn_list),
        np.ascontiguousarray(crd),
        iscreen
    ).base
