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
import numpy as np
import cython

DTYPE = np.intc

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.embedsignature(True)
def retain_full_residues_cy(int[:] idx, int[:] residues_array):
    '''
    Given an array `idx`, with 1 standing for `selected atom`
    and 0 standing for `unselected atom`, and an array `residues_array`
    with entries corresponding to the beginning of a residue in `idx`,
    yields a new array `new_idx` with entries 1 for all atoms of each residue
    for which at least on atom is selected in the original `idx` array.

    Example:
        >>> idx = np.array([1, 0, 0, 0, 0, 0])
        >>> residues_array = np.array([0, 2])
        >>> new_idx = retain_full_residues(idx, residues_array)
        >>> print(new_idx)
            [1, 1, 0, 0, 0, 0]
    '''
    cdef Py_ssize_t idx_max = idx.shape[0]
    _new_idx = np.zeros((idx_max), dtype=DTYPE)
    cdef int[:] new_idx = _new_idx

    cdef Py_ssize_t n_resids = residues_array.shape[0] - 1
    cdef Py_ssize_t i, j, k
    cdef int start
    cdef int stop
    cdef int val

    for i in range(0, n_resids):
        val = 0
        start = residues_array[i]
        stop = residues_array[i+1]
        for j in range(start, stop):
            val += idx[j]
        if val > 0.5:
            for k in range(start, stop):
                new_idx[k] = 1

    return new_idx.base
