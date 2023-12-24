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


@njit(parallel=True, error_model='numpy', cache=True)
def _tdma_parallel_core(ldiag: np.ndarray[np.floating],
                        cdiag: np.ndarray[np.floating],
                        rdiag: np.ndarray[np.floating],
                        rhs: np.ndarray[np.floating]) -> np.ndarray[np.floating]:

    out = np.zeros_like(rhs)

    n = rhs.shape[1]
    w = np.zeros(n, ldiag.dtype)
    b1 = np.zeros(n, ldiag.dtype)

    # prepare tdma

    b1[0] = cdiag[0]
    for i in range(1, n):
        w[i] = ldiag[i]/b1[i-1]
        b1[i] = cdiag[i]-w[i]*rdiag[i-1]

    for j in prange(rhs.shape[0]):
        d = np.zeros(n, ldiag.dtype)
        d[0] = rhs[j, 0]
        for i in range(1, n):
            d[i] = rhs[j, i]-w[i]*d[i-1]

        out[j, n-1] = d[n-1]/b1[n-1]
        for i in range(n-2, -1, -1):
            out[j, i] = (d[i]-rdiag[i]*out[j, i+1])/b1[i]

    return out
