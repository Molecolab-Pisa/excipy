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
cimport numpy as np
import numpy as np
import cython

cdef _map_polarizable_atoms(np.int64_t[:] cut_mask, str count_as="fortran"):
    # total number of atoms
    cdef int num_atoms = cut_mask.shape[0]

    # set how to count
    cdef int c
    if count_as == "fortran":
        c = 1
    else:
        c = 0

    # goes from an index running over all atoms to an index
    # running over over the pol atoms
    _pol_map = np.zeros(num_atoms+c, dtype=np.int64) -1 + c
    cdef np.int64_t[:] pol_map = _pol_map

    cdef int num_pol_atoms = 0
    cdef int i
    cdef int f = 0
    for i in range(num_atoms):
        if cut_mask[i] == 1:
            num_pol_atoms += 1
            pol_map[i+c] = f + c
            f += 1


    _pol_idx = np.empty(num_pol_atoms, dtype=np.int64)
    cdef np.int64_t[:] pol_idx = _pol_idx

    cdef int e = 0
    for i in range(num_atoms):
        if cut_mask[i] == 1:
            pol_idx[e] = i
            e += 1


    return pol_map, num_pol_atoms, pol_idx


def map_polarizable_atoms_cy(cut_mask, str count_as="fortran"):
    pol_map, num_pol_atoms, pol_idx = _map_polarizable_atoms(cut_mask.astype(np.int64), count_as)
    return pol_map.base, num_pol_atoms, pol_idx.base
