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
from libc.math cimport sqrt

@cython.wraparound(False)
@cython.boundscheck(False)
cdef double[:,::1] _distances_diffmask(
    double[:,::1] source_coords,
    double[:,::1] ext_coords,
    double[:,::1] dist1,
    np.int32_t[::1] cut0_mask,
    np.int32_t[::1] cut1_mask,
    np.int32_t[::1] cut2_mask,
):
    # final number of external coordinates
    cdef int num_cut2 = 0
    cdef int i
    for i in range(cut2_mask.shape[0]):
        if cut2_mask[i] == 1:
            num_cut2 += 1

    # array of distances to be returned
    _dist = np.empty((num_cut2, source_coords.shape[0]), dtype=np.float64)
    cdef double[:,::1] dist = _dist

    # index of the row in dist, goes up when a 1 is
    # encountered in cut2_mask
    cdef int ii = -1

    # index of the row in dist1, goes up when a 1 is
    # encountered in cut0_mask
    cdef int jj = -1

    # loop variables, j for source, k on cartesian
    cdef int j, k

    # distance, needed when computing the missing distances
    # in dist1
    cdef double d = 0.0

    for i in range(ext_coords.shape[0]):
        # if an atom is included in the first cut (we have a distance)
        # we must update the index
        if cut0_mask[i] == 1:
            jj += 1

        # check if we need to provide a distance
        if cut2_mask[i] == 1:
            ii += 1

            # try to fetch the distance if already computed
            if cut1_mask[i] == 1:
                for j in range(source_coords.shape[0]):
                    dist[ii,j] = dist1[jj,j]
            else:
                # we must compute the missing distances
                for j in range(source_coords.shape[0]):
                    d = 0.0
                    for k in range(3):
                        d += (source_coords[j,k] - ext_coords[i,k])**2
                    dist[ii,j] = sqrt(d)

    return dist

@cython.embedsignature(True)
def distances_diffmask_cy(double[:,:] source_coords, double[:,:] ext_coords, double[:,:] dist1, cut0_mask, cut1_mask, cut2_mask):
    """
    Computes only the missing distances in dist1 for external atoms
    that are included in cut2_mask but were not included in cut1_mask
    (e.g. cut1_mask cuts through the molecule, and cut2_mask keeps the
    molecules intact), reusing the distances computed in a first stage
    using cut0_mask for the external part (e.g., cut0_mask could come
    from a square cut of the box).

    Parameters
    ----------
    source_coords: np.ndarray, shape (num_source, 3)
        source coordinates
    ext_coords: np.ndarray, shape (num_ext, 3)
        external coordinates
    dist1: np.ndarray, shape (num_cut0, num_source)
        distances computed using the cut0_mask to select the external
        coordinates (tipically the mask from cut_box)
    cut0_mask: np.ndarray, shape (num_cut0)
        array of 1 if an external atom is selected in the first cut and
        0 otherwise
    cut1_mask: np.ndarray, shape (num_cut1)
        array of 1 if an external atom is selected in the second cut and
        0 otherwise
    cut2_mask: np.ndarray, shape (num_cut2)
        array of 1 if an external atom is selected in the final cut and
        0 otherwise

    Returns
    -------
    dist: np.ndarray, shape (num_cut2, num_source)
        distances between external coordinates and source coordinates
        when selecting atoms according to cut2_mask.
    """
    return _distances_diffmask(
        np.ascontiguousarray(source_coords),
        np.ascontiguousarray(ext_coords),
        np.ascontiguousarray(dist1),
        cut0_mask.astype(np.int32),
        cut1_mask.astype(np.int32),
        cut2_mask.astype(np.int32),
    )
