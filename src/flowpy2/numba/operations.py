import numpy as np
from numba import njit, prange


def tdma_solve(ldiag, cdiag, rdiag, rhs):
    if len(ldiag) != len(rdiag):
        raise ValueError("Left and right diagonals must be the same size")

    if len(ldiag) != len(cdiag):
        raise ValueError("left and right diagonals should be equal"
                         " length compared with cdiag")

    if len(ldiag) != rhs.shape[-1]:
        raise ValueError("Diagonals should be the same length as the "
                         "right-most dimension of the rhs array")

    if rhs.ndim != 2:
        raise ValueError("This method only works on two dimensions")

    return _tdma_parallel_core(ldiag, cdiag, rdiag, rhs)


_tdma_signature = ["f8[:,:](f8[:],f8[:],f8[:],f8[:,:])",
                   "f4[:,:](f4[:],f4[:],f4[:],f4[:,:])"]


@njit(_tdma_signature, parallel=True, error_model='numpy', cache=True)
def _tdma_parallel_core(ldiag: np.ndarray,
                        cdiag: np.ndarray,
                        rdiag: np.ndarray,
                        rhs: np.ndarray) -> np.ndarray:

    out = np.zeros_like(rhs)

    n = rhs.shape[1]
    w = np.zeros(n, ldiag.dtype)
    b1 = np.zeros(n, ldiag.dtype)

    # prepare tdma

    b1[0] = cdiag[0]
    for i in range(1, n):
        w[i] = ldiag[i]/b1[i-1]
        b1[i] = -w[i]*rdiag[i-1] + cdiag[i]

    ib1 = 1./b1
    for j in prange(rhs.shape[0]):
        d = np.zeros(n, ldiag.dtype)
        d[0] = rhs[j, 0]
        for i in range(1, n):
            d[i] = -d[i-1]*w[i] + rhs[j, i]

        out[j, n-1] = d[n-1]*ib1[n-1]
        for i in range(n-2, -1, -1):
            out[j, i] = (-out[j, i+1]*rdiag[i] + d[i])*ib1[i]

    return out
