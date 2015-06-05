#!/usr/bin/env python
import numpy as np
cimport numpy as np
import cython

# profiling suggests removing bounds checking doesn't
#   increase performance very significantly
# @cython.boundscheck(False)
# @cython.wraparound(False)
def _unique(np.ndarray[np.int32_t,ndim=1] arr not None):
  cdef int imax = arr.size
  cdef np.ndarray[np.int32_t,ndim=1] unique = np.empty(imax, dtype=np.int32)
  cdef int i, j
  cdef int n_unq = 0
  cdef int value
  for i in range(imax):
    value = arr[i]
    for j in range(n_unq+1):
      if unique[j] == value:
        break
      if j == n_unq:
        unique[n_unq] = value
        n_unq += 1
      
  return unique[:n_unq]


