import numpy as np
from numba import njit, prange
from .operations import tdma_solve
from ._utils import log_jit

_gradient_dx_signature = ['f8[:,:](f8[:,:], f8)',
                          'f4[:,:](f4[:,:], f4)']

_gradient_varx_signature = ['f8[:,:](f8[:,:], f8[:])',
                            'f4[:,:](f4[:,:], f4[:])']


@njit(parallel=True,
      error_model='numpy', cache=True)
def gradient1_order2_dx(array: np.ndarray, dx: float):
    gradient = np.zeros_like(array)
    idx = 1./dx
    for i in prange(array.shape[0]):

        gradient[i, 0] = idx * \
            (-1.5*array[i, 0] + 2.*array[i, 1] - 0.5*array[i, 2])

        for j in range(1, array.shape[-1]-1):
            gradient[i, j] = 0.5*idx*(- array[i, j-1] + array[i, j+1])

        gradient[i, -1] = idx * \
            (0.5*array[i, -3] - 2.*array[i, -2] + 1.5*array[i, -1])

    return gradient


@njit( parallel=True, error_model='numpy', cache=True)
def gradient1_order2_var_x(array: np.ndarray, x: float):
    gradient = np.zeros_like(array)

    for i in prange(array.shape[0]):
        dx1 = x[1] - x[0]
        dx2 = x[2] - x[1]

        a = -(2*dx1 + dx2)/(dx1*(dx1 + dx2))
        b = (dx1 + dx2)/(dx1*dx2)
        c = -dx1/(dx2*(dx1 + dx2))

        gradient[i, 0] = a*array[i, 0] + b*array[i, 1] + c*array[i, 2]

        for j in range(1, array.shape[-1]-1):
            dx1 = x[j] - x[j-1]
            dx2 = x[j+1] - x[j]

            a = -dx2/(dx1*(dx1 + dx2))
            b = (-dx1 + dx2)/(dx1*dx2)
            c = dx1/(dx2*(dx1 + dx2))
            gradient[i, j] = a*array[i, j-1] + b*array[i, j] + c*array[i, j+1]

        dx1 = x[-2] - x[-3]
        dx2 = x[-1] - x[-2]

        a = dx2/(dx1*(dx1 + dx2))
        b = -(dx1 + dx2)/(dx1*dx2)
        c = (2*dx2 + dx1)/(dx2*(dx1 + dx2))

        gradient[i, -1] = a*array[i, -3] + b*array[i, -2] + c*array[i, -1]
    return gradient


@njit(parallel=True, error_model='numpy', cache=True)
def gradient2_order2_dx(array: np.ndarray, dx: float):
    gradient = np.zeros_like(array)
    idx2 = 1./(dx*dx)
    a = idx2
    b = -2.*idx2
    c = idx2
    for i in prange(array.shape[0]):
        gradient[i, 0] = a*array[i, 0] + b*array[i, 1] + c*array[i, 2]

        for j in range(1, array.shape[-1]-1):
            gradient[i, j] = a*array[i, j-1] + b*array[i, j] + c*array[i, j+1]

        gradient[i, -1] = a*array[i, -3] + b*array[i, -2] + c*array[i, -1]

    return gradient


@njit(parallel=True,
      error_model='numpy', cache=True)
def gradient2_order2_var_x(array: np.ndarray, x: float):
    gradient = np.zeros_like(array)
    for i in prange(array.shape[0]):
        dx1 = x[1] - x[0]
        dx2 = x[2] - x[1]

        a = 2./(dx1*(dx1 + dx2))
        b = -2/(dx1*dx2)
        c = 2/(dx2*(dx1 + dx2))

        gradient[i, 0] = a*array[i, 0] + b*array[i, 1] + c*array[i, 2]

        for j in range(1, array.shape[-1]-1):
            dx1 = x[j] - x[j-1]
            dx2 = x[j+1] - x[j]

            a = 2./(dx1*(dx1 + dx2))
            b = -2/(dx1*dx2)
            c = 2/(dx2*(dx1 + dx2))
            gradient[i, j] = a*array[i, j-1] + b*array[i, j] + c*array[i, j+1]

        dx1 = x[-3] - x[-2]
        dx2 = x[-2] - x[-1]

        a = 2./(dx1*(dx1 + dx2))
        b = -2/(dx1*dx2)
        c = 2/(dx2*(dx1 + dx2))

        gradient[i, -1] = a*array[i, -3] + b*array[i, -2] + c*array[i, -1]
    return gradient


@njit(parallel=True,
      error_model='numpy', cache=True)
def _rhs_gradient1_order6_dx(array: np.ndarray, dx: float):
    idx = 1./dx
    a = 7.*idx/9.
    b = 1.*idx/36.

    rhs = np.zeros_like(array)
    rhs[:, 0] = idx*(-2.5*array[:, 0] + 2.*array[:, 1] + 0.5*array[:, 2])
    rhs[:, 1] = 0.75*idx*(array[:, 2] - array[:, 0])

    # boundary at n, n-1

    rhs[:, -1] = idx*(2.5*array[:, -1] - 2.*array[:, -2]
                      - 0.5*array[:, -3])
    rhs[:, -2] = 0.75*idx*(array[:, -1] - array[:, -3])

    # middle
    rhs[:, 2:-2] = a*(array[:, 3:-1] - array[:, 1:-3]) \
        + b*(array[:, 4:] - array[:, :-4])

    return rhs


def gradient1_order6_dx(array: np.ndarray, dx: float):

    ldiag = np.zeros(array.shape[-1])
    cdiag = np.ones(array.shape[-1])
    rdiag = np.zeros(array.shape[-1])

    ldiag[1] = 0.25
    ldiag[2:-2] = 1/3
    ldiag[-2] = 0.25
    ldiag[-1] = 2

    rdiag[0] = 2
    rdiag[1] = 0.25
    rdiag[2:-2] = 1/3
    rdiag[-2] = 0.25

    rhs = _rhs_gradient1_order6_dx(array, dx)

    return tdma_solve(ldiag, cdiag, rdiag, rhs)


@njit(parallel=True,
      error_model='numpy', cache=True)
def _rhs_gradient2_order6_dx(array: np.ndarray, dx: float):
    idx2 = 1./(dx*dx)
    a = 12.*idx2/11.
    b = 3.*idx2/44

    rhs = np.zeros_like(array)
    rhs[:, 0] = idx2*(13.*array[:, 0] + -27.*array[:, 1] +
                      15.*array[:, 2] - array[:, 3])
    rhs[:, 1] = 1.2*idx2*(array[:, 0] - 2.*array[:, 1] + array[:, 2])

    # boundary at n, n-1

    rhs[:, -1] = idx2*(13.*array[:, -1] - 27.*array[:, -2] +
                       15.*array[:, -3] - array[:, -4])
    rhs[:, -2] = 1.2*idx2*(array[:, -3] - 2.*array[:, -2] + array[:, -1])

    # middle
    rhs[:, 2:-2] = a*(array[:, 3:-1] - 2.*array[:, 2:-2] + array[:, 1:-3]) \
        + b*(array[:, 4:] - 2.*array[:, 2:-2] + array[:, :-4])

    return rhs


def gradient2_order6_dx(array: np.ndarray, dx: float):
    ldiag = np.zeros(array.shape[-1])
    cdiag = np.ones(array.shape[-1])
    rdiag = np.zeros(array.shape[-1])

    ldiag[1] = 0.1
    ldiag[2:-2] = 2./11.
    ldiag[-2] = 0.1
    ldiag[-1] = 11.

    rdiag[0] = 11
    rdiag[1] = 0.1
    rdiag[2:-2] = 2./11.
    rdiag[-2] = 0.1

    rhs = _rhs_gradient2_order6_dx(array, dx)

    return tdma_solve(ldiag, cdiag, rdiag, rhs)


# @njit(_gradient_varx_signature, parallel=True,
#       error_model='numpy', cache=True)
# def _rhs_gradient1_order6_var_x(array: np.ndarray, dx_array: np.ndarray):

#     rhs = np.zeros_like(array)
#     # point 0
#     a = None
#     b = None
#     c = None
#     rhs[:, 0] = a*array[:, 0] + b*array[:, 1] + c*array[:, 2]

#     a = None
#     b = None
#     c = None
#     rhs[:, 1] = a*array[:, 0] + b*array[:, 1] + c*array[:, 2]

#     # point n

#     a = None
#     b = None
#     c = None
#     rhs[:, -1] = a*array[:, -1] + b*array[:, -2] + c*array[:, -3]

#     # point n-1
#     a = None
#     b = None
#     c = None

#     rhs[:, -2] = a*array[:, -3] + b*array[:, -2] + array[:, -1]

#     # middle
#     a = np.array(array.shape[-1]-2)
#     b = np.array(array.shape[-1]-2)
#     c = np.array(array.shape[-1]-2)
#     d = np.array(array.shape[-1]-2)
#     e = np.array(array.shape[-1]-2)

#     rhs[:, 2:-2] = a*array[:, :-4] + b*array[:, 1:-3] + c*array[:, 2:-2] \
#         + d*array[:, 3:-1] + e*array[:, 4:]

#     return rhs


# def gradient1_order6_var_x(array: np.ndarray, x: np.ndarray):
#     ldiag = np.zeros(array.shape[-1])
#     cdiag = np.ones(array.shape[-1])
#     rdiag = np.zeros(array.shape[-1])

#     dx_array = np.zeros(array.shape[-1]-1)

#     # update
#     ldiag[1] = 0.25
#     ldiag[2:-2] = 1/3
#     ldiag[-2] = 0.25
#     ldiag[-1] = 2

#     rdiag[0] = 2
#     rdiag[1] = 0.25
#     rdiag[2:-2] = 1/3
#     rdiag[-2] = 0.25

#     rhs = _rhs_gradient1_order6_var_x(array, dx_array)

#     return tdma_solve(ldiag, cdiag, rdiag, rhs)


# @njit(_gradient_varx_signature, parallel=True, error_model='numpy', cache=True)
# def _rhs_gradient2_order6_var_x(array: np.ndarray, dx_array: np.ndarray):

#     rhs = np.zeros_like(array)
#     # point 0
#     a = None
#     b = None
#     c = None
#     rhs[:, 0] = a*array[:, 0] + b*array[:, 1] + c*array[:, 2]

#     a = None
#     b = None
#     c = None
#     rhs[:, 1] = a*array[:, 0] + b*array[:, 1] + c*array[:, 2]

#     # point n

#     a = None
#     b = None
#     c = None
#     rhs[:, -1] = a*array[:, -1] + b*array[:, -2] + c*array[:, -3]

#     # point n-1
#     a = None
#     b = None
#     c = None

#     rhs[:, -2] = a*array[:, -3] + b*array[:, -2] + array[:, -1]

#     # middle
#     a = np.array(array.shape[-1]-2)
#     b = np.array(array.shape[-1]-2)
#     c = np.array(array.shape[-1]-2)
#     d = np.array(array.shape[-1]-2)
#     e = np.array(array.shape[-1]-2)

#     rhs[:, 2:-2] = a*array[:, :-4] + b*array[:, 1:-3] + c*array[:, 2:-2] \
#         + d*array[:, 3:-1] + e*array[:, 4:]

#     return rhs


# def gradient2_order6_var_x(array: np.ndarray, x: np.ndarray):
#     ldiag = np.zeros(array.shape[-1])
#     cdiag = np.ones(array.shape[-1])
#     rdiag = np.zeros(array.shape[-1])

#     dx_array = np.zeros(array.shape[-1]-1)

#     # update
#     ldiag[1] = 0.25
#     ldiag[2:-2] = 1/3
#     ldiag[-2] = 0.25
#     ldiag[-1] = 2

#     rdiag[0] = 2
#     rdiag[1] = 0.25
#     rdiag[2:-2] = 1/3
#     rdiag[-2] = 0.25

#     rhs = _rhs_gradient2_order6_var_x(array, dx_array)

#     return tdma_solve(ldiag, cdiag, rdiag, rhs)


log_jit(gradient1_order2_dx)
log_jit(gradient1_order2_var_x)
log_jit(gradient2_order2_dx)
log_jit(gradient2_order2_var_x)
log_jit(_rhs_gradient1_order6_dx)
log_jit(_rhs_gradient2_order6_dx)
# log_jit(_rhs_gradient1_order6_var_x)
# log_jit(_rhs_gradient2_order6_var_x)
